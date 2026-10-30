"""Servicio de recuperación y generación de respuestas (RAG).

Coordina la búsqueda semántica de fragmentos en ChromaDB y la formulación de
respuestas con citas precisas de documento y página. Implementa una estrategia
local sin costo y soporte opcional para OpenAI si se proporciona API key.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings, get_settings
from app.preguntas.schemas import Cita, PreguntaRequest, RespuestaResponse
from app.vector_store import create_vector_collection


class PreguntasService:
    """Servicio que ejecuta la búsqueda vectorial y sintetiza la respuesta con citas."""

    def __init__(self, settings: Optional[Settings] = None):
        """Inicializa el servicio de preguntas.

        Args:
            settings: Configuración global inyectada o por defecto.
        """
        self.settings = settings or get_settings()
        self._init_chroma()

    def _init_chroma(self) -> None:
        """Conecta con la colección vectorial persistente."""
        self.collection = create_vector_collection(self.settings)

    def retrieve_relevant_chunks(
        self, pregunta: str, top_k: int = 3, documento_id: Optional[str] = None
    ) -> List[Cita]:
        """Consulta ChromaDB para encontrar los fragmentos más afines a la pregunta.

        Args:
            pregunta: Consulta del usuario en lenguaje natural.
            top_k: Número de fragmentos a recuperar.
            documento_id: Filtro opcional por ID de documento específico.

        Returns:
            List[Cita]: Lista ordenada de citas con metadatos de documento y página.
        """
        count = self.collection.count()
        if count == 0:
            return []

        where_filter: Optional[Dict[str, Any]] = None
        if documento_id:
            where_filter = {"documento_id": documento_id}

        # Ajustar n_results si hay menos fragmentos indexados que top_k
        n_results = min(top_k, count)
        if n_results <= 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[pregunta],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        citas: List[Cita] = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return citas

        documents_list = results["documents"][0]
        metadatas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]

        for i, text in enumerate(documents_list):
            meta = metadatas_list[i] if i < len(metadatas_list) else {}
            dist = distances_list[i] if distances_list and i < len(distances_list) else None

            # Calcular score de relevancia normalizado a partir de distancia
            relevancia: Optional[float] = None
            if dist is not None:
                relevancia = round(max(0.0, 1.0 - float(dist)), 4)

            citas.append(
                Cita(
                    pagina=int(meta.get("pagina", 1)),
                    texto=text,
                    documento=str(meta.get("nombre_archivo", "apuntes.pdf")),
                    documento_id=meta.get("documento_id"),
                    posicion=meta.get("posicion"),
                    relevancia=relevancia,
                )
            )

        return citas

    def _generate_openai_answer(self, pregunta: str, citas: List[Cita]) -> Optional[str]:
        """Genera respuesta enriquecida utilizando la API de OpenAI si está configurada.

        Args:
            pregunta: Pregunta del usuario.
            citas: Lista de fragmentos y páginas obtenidos.

        Returns:
            Optional[str]: Respuesta generada o None si falla.
        """
        if not self.settings.openai_api_key:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.settings.openai_api_key)

            context_snippets = "\n\n".join([
                f"[Documento: {c.documento} | Página: {c.pagina}]\n{c.texto}"
                for c in citas
            ])

            system_prompt = (
                "Eres un tutor académico universitario del sistema 'Academic'. "
                "Responde a la pregunta del estudiante basándote estrictamente en el contexto provisto de sus apuntes. "
                "Cita explícitamente el nombre del documento y el número de página en cada afirmación relevante. "
                "Si la información no está en el contexto, indícalo amablemente sin inventar datos."
            )

            user_prompt = (
                f"Contexto de apuntes:\n{context_snippets}\n\n"
                f"Pregunta del estudiante: {pregunta}"
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )

            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
        except Exception:
            pass

        return None

    def _generate_local_answer(self, pregunta: str, citas: List[Cita]) -> str:
        """Sintetiza la respuesta directamente a partir de las citas más relevantes sin requerir API externa.

        Estrategia limpia y autosuficiente: formatea los fragmentos recuperados con sus
        páginas y documentos de procedencia para lectura directa del estudiante.

        Args:
            pregunta: Pregunta consultada.
            citas: Citas encontradas en ChromaDB.

        Returns:
            str: Respuesta articulada y legible con citas exactas.
        """
        if not citas:
            return (
                "No se encontró información relevante en los apuntes indexados para responder "
                "a la pregunta formulada. Asegúrate de haber subido el PDF correspondiente."
            )

        lineas: List[str] = [
            f"Basado en los apuntes académicos indexados, se encontró la siguiente información relevante para responder a: \"{pregunta}\":\n"
        ]

        for idx, cita in enumerate(citas, 1):
            lineas.append(
                f"[{idx}] {cita.documento} (Página {cita.pagina}):\n"
                f"\"{cita.texto}\"\n"
            )

        lineas.append(
            "Consulta las citas adjuntas para revisar los detalles completos en cada página del documento."
        )

        return "\n".join(lineas)

    def responder(self, request: PreguntaRequest) -> RespuestaResponse:
        """Atiende la solicitud de consulta completa: búsqueda semántica y generación de respuesta.

        Args:
            request: Parámetros de la pregunta (texto, documento opcional, top_k).

        Returns:
            RespuestaResponse: Objeto con la respuesta y lista estructurada de citas.
        """
        citas = self.retrieve_relevant_chunks(
            pregunta=request.pregunta,
            top_k=request.top_k,
            documento_id=request.documento_id,
        )

        if not citas:
            return RespuestaResponse(
                pregunta=request.pregunta,
                respuesta=(
                    "No se encontró información relevante en los apuntes indexados para responder "
                    "a la pregunta. Verifica que el apunte esté subido correctamente."
                ),
                citas=[],
                total_citas=0,
                proveedor_llm=self.settings.llm_provider,
            )

        # Determinar generador según configuración
        proveedor_utilizado = "local"
        respuesta_texto: Optional[str] = None

        if self.settings.llm_provider.lower() == "openai" and self.settings.openai_api_key:
            respuesta_texto = self._generate_openai_answer(request.pregunta, citas)
            if respuesta_texto:
                proveedor_utilizado = "openai"

        if not respuesta_texto:
            respuesta_texto = self._generate_local_answer(request.pregunta, citas)
            proveedor_utilizado = "local"

        return RespuestaResponse(
            pregunta=request.pregunta,
            respuesta=respuesta_texto,
            citas=citas,
            total_citas=len(citas),
            proveedor_llm=proveedor_utilizado,
        )
