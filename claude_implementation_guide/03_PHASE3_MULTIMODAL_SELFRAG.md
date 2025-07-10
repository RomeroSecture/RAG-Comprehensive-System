# Phase 3: Multimodal & Self-RAG Implementation Guide

## 📋 Objetivo de la Fase

Implementar capacidades multimodales (texto + imagen) y Self-RAG con mecanismos de auto-reflexión, corrección dinámica y procesamiento en tiempo real, incluyendo un dashboard interactivo.

## 🎯 Entregables Clave

1. Procesamiento multimodal de documentos (PDF con imágenes, diagramas)
2. Vision-Language Models integration (GPT-4V, CLIP)
3. Self-RAG con confidence scoring y auto-corrección
4. Corrective RAG con evaluación iterativa
5. Long RAG para documentos extensos
6. Dashboard Next.js con visualización en tiempo real
7. WebSocket API para streaming responses

## 📅 Timeline Detallado

### Semana 13-14: Multimodal RAG

#### Día 1-3: Vision-Language Model Integration

```python
# src/infrastructure/external/multimodal/vision_service.py
import base64
from typing import List, Tuple
from PIL import Image
import io

class VisionLanguageService:
    def __init__(self, 
                 openai_client: AsyncOpenAI,
                 model: str = "gpt-4-vision-preview"):
        self.client = openai_client
        self.model = model
    
    async def analyze_image(self, 
                          image_path: str, 
                          prompt: str = None) -> Dict[str, any]:
        # Cargar y encodear imagen
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prompt por defecto para extracción
        if not prompt:
            prompt = """Analyze this image and provide:
            1. A detailed description
            2. Any text found in the image
            3. Key visual elements (charts, diagrams, tables)
            4. Relevant metadata for search indexing"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return {
            "description": response.choices[0].message.content,
            "image_path": image_path,
            "model": self.model
        }
    
    async def extract_text_from_image(self, image_path: str) -> str:
        """OCR + Vision model para mejor extracción de texto"""
        # Primero intentar OCR tradicional
        ocr_text = await self._perform_ocr(image_path)
        
        # Luego mejorar con vision model
        vision_prompt = f"""
        Extract all text from this image. 
        OCR detected: {ocr_text[:200]}...
        Please provide the complete and corrected text.
        """
        
        result = await self.analyze_image(image_path, vision_prompt)
        return result["description"]
```

#### Día 4-6: Multimodal Embeddings

```python
# src/infrastructure/external/multimodal/multimodal_embedder.py
import clip
import torch
from transformers import CLIPModel, CLIPProcessor

class MultimodalEmbeddingService:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    async def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        return text_features.cpu().numpy()[0]
    
    async def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features.cpu().numpy()[0]
    
    async def embed_multimodal(self, 
                             text: str, 
                             image_paths: List[str]) -> np.ndarray:
        """Crear embedding combinado de texto e imágenes"""
        # Embed texto
        text_embedding = await self.embed_text(text)
        
        # Embed imágenes
        image_embeddings = []
        for img_path in image_paths:
            img_emb = await self.embed_image(img_path)
            image_embeddings.append(img_emb)
        
        if image_embeddings:
            # Promediar embeddings de imágenes
            avg_image_embedding = np.mean(image_embeddings, axis=0)
            
            # Combinar con pesos
            combined = 0.7 * text_embedding + 0.3 * avg_image_embedding
            return combined / np.linalg.norm(combined)
        
        return text_embedding
```

#### Día 7-9: Multimodal Document Processing

```python
# src/application/services/multimodal_processor.py
class MultimodalDocumentProcessor:
    def __init__(self,
                 vision_service: VisionLanguageService,
                 multimodal_embedder: MultimodalEmbeddingService,
                 text_parser: TextParser,
                 image_extractor: ImageExtractor):
        self.vision_service = vision_service
        self.multimodal_embedder = multimodal_embedder
        self.text_parser = text_parser
        self.image_extractor = image_extractor
    
    async def process_multimodal_document(self, 
                                        file_path: str) -> MultimodalDocument:
        # 1. Extraer texto y imágenes
        text_content = await self.text_parser.parse(file_path)
        images = await self.image_extractor.extract_images(file_path)
        
        # 2. Procesar cada imagen
        image_analyses = []
        for img_path in images:
            analysis = await self.vision_service.analyze_image(img_path)
            image_analyses.append({
                "path": img_path,
                "description": analysis["description"],
                "embedding": await self.multimodal_embedder.embed_image(img_path)
            })
        
        # 3. Crear chunks considerando imágenes
        chunks = await self._create_multimodal_chunks(
            text_content, image_analyses
        )
        
        # 4. Generar embeddings multimodales
        for chunk in chunks:
            if chunk.associated_images:
                chunk.embedding = await self.multimodal_embedder.embed_multimodal(
                    chunk.content,
                    chunk.associated_images
                )
            else:
                chunk.embedding = await self.multimodal_embedder.embed_text(
                    chunk.content
                )
        
        return MultimodalDocument(
            text_content=text_content,
            images=image_analyses,
            chunks=chunks
        )
    
    async def _create_multimodal_chunks(self,
                                      text: str,
                                      images: List[Dict]) -> List[MultimodalChunk]:
        """Crear chunks que incluyan referencias a imágenes cercanas"""
        # Lógica para asociar imágenes con chunks de texto relevantes
        pass
```

#### Día 10-12: Cross-Modal Search

```python
# src/infrastructure/persistence/vector_stores/multimodal_vector_store.py
class MultimodalVectorStore:
    def __init__(self,
                 text_store: VectorStoreRepository,
                 image_store: VectorStoreRepository):
        self.text_store = text_store
        self.image_store = image_store
    
    async def cross_modal_search(self,
                               query: str,
                               query_image: Optional[str] = None,
                               k: int = 10) -> List[MultimodalSearchResult]:
        results = []
        
        # 1. Si hay query de texto, buscar en ambos stores
        if query:
            # Buscar texto similar
            text_embedding = await self.embedder.embed_text(query)
            text_results = await self.text_store.search(text_embedding, k)
            
            # Buscar imágenes relevantes usando CLIP
            image_results = await self.image_store.search(text_embedding, k)
            
            results.extend(self._format_results(text_results, "text"))
            results.extend(self._format_results(image_results, "image"))
        
        # 2. Si hay imagen de query, búsqueda inversa
        if query_image:
            image_embedding = await self.embedder.embed_image(query_image)
            
            # Buscar imágenes similares
            similar_images = await self.image_store.search(image_embedding, k)
            
            # Buscar texto relacionado
            related_text = await self.text_store.search(image_embedding, k)
            
            results.extend(self._format_results(similar_images, "image"))
            results.extend(self._format_results(related_text, "text"))
        
        # 3. Fusionar y rankear resultados
        return self._fuse_multimodal_results(results, k)
```

### Semana 15-16: Self-RAG & Corrective RAG

#### Día 13-15: Self-RAG Implementation

```python
# src/application/services/self_rag_orchestrator.py
class SelfRAGOrchestrator:
    def __init__(self,
                 query_analyzer: QueryAnalyzerService,
                 retrieval_critic: RetrievalCritic,
                 adaptive_retriever: AdaptiveRetriever,
                 generator: LLMGenerator,
                 response_evaluator: ResponseEvaluator):
        self.query_analyzer = query_analyzer
        self.retrieval_critic = retrieval_critic
        self.adaptive_retriever = adaptive_retriever
        self.generator = generator
        self.response_evaluator = response_evaluator
        self.max_iterations = 3
    
    async def self_reflective_rag(self, query: Query) -> SelfRAGResult:
        # 1. Analizar si se necesita retrieval
        retrieval_decision = await self._decide_retrieval_need(query)
        
        if not retrieval_decision.needs_retrieval:
            # Generación directa sin retrieval
            response = await self.generator.generate_direct(query.text)
            return SelfRAGResult(
                response=response,
                confidence_score=0.9,
                retrieval_attempts=0
            )
        
        # 2. Retrieval iterativo con auto-crítica
        context_documents = []
        reformulated_queries = []
        confidence_scores = []
        
        for iteration in range(self.max_iterations):
            # Retrieve
            retrieved = await self.adaptive_retriever.retrieve(query)
            
            # Evaluar calidad del retrieval
            retrieval_quality = await self.retrieval_critic.evaluate(
                query, retrieved
            )
            
            confidence_scores.append(retrieval_quality.score)
            
            if retrieval_quality.is_sufficient():
                context_documents.extend(retrieved.documents)
                break
            else:
                # Reformular query basado en feedback
                feedback = retrieval_quality.get_feedback()
                reformulated = await self.query_analyzer.reformulate(
                    query, feedback
                )
                reformulated_queries.append(reformulated.text)
                query = reformulated
        
        # 3. Generación con verificación
        response = await self._generate_with_verification(
            query, context_documents
        )
        
        # 4. Evaluación final de confianza
        final_confidence = await self.response_evaluator.assess_confidence(
            response, context_documents
        )
        
        return SelfRAGResult(
            query_id=query.id,
            documents=context_documents,
            response=response,
            confidence_score=final_confidence,
            retrieval_attempts=iteration + 1,
            reformulated_queries=reformulated_queries,
            confidence_history=confidence_scores
        )
    
    async def _decide_retrieval_need(self, query: Query) -> RetrievalDecision:
        """Decidir si la query necesita retrieval externo"""
        prompt = f"""
        Query: {query.text}
        
        Determine if this query requires external information retrieval.
        Consider:
        1. Is this asking for specific facts or data?
        2. Does it require current information?
        3. Can it be answered with general knowledge?
        
        Output: NEEDS_RETRIEVAL or NO_RETRIEVAL with reasoning
        """
        
        decision = await self.generator.analyze(prompt)
        return RetrievalDecision.parse(decision)
```

#### Día 16-18: Corrective RAG Implementation

```python
# src/application/services/corrective_rag_service.py
class CorrectiveRAGService:
    def __init__(self,
                 retriever: RetrievalService,
                 fact_checker: FactCheckingService,
                 corrector: CorrectionService):
        self.retriever = retriever
        self.fact_checker = fact_checker
        self.corrector = corrector
    
    async def retrieve_and_correct(self, query: Query) -> CorrectedRetrievalResult:
        # 1. Initial retrieval
        initial_results = await self.retriever.retrieve(query)
        
        # 2. Verificar factualidad de cada documento
        verification_results = []
        for doc in initial_results.documents:
            is_factual = await self.fact_checker.verify_document(
                doc, query.text
            )
            verification_results.append({
                "document": doc,
                "is_factual": is_factual,
                "confidence": is_factual.confidence
            })
        
        # 3. Filtrar documentos no factuales
        factual_docs = [
            v["document"] for v in verification_results 
            if v["is_factual"].is_valid
        ]
        
        # 4. Si no hay suficientes documentos válidos, buscar más
        if len(factual_docs) < query.max_results * 0.5:
            # Expandir búsqueda con estrategias alternativas
            additional_docs = await self._expand_search(query)
            factual_docs.extend(additional_docs)
        
        # 5. Corregir información si es necesario
        corrected_docs = []
        for doc in factual_docs:
            if doc.needs_correction:
                corrected = await self.corrector.correct_document(doc)
                corrected_docs.append(corrected)
            else:
                corrected_docs.append(doc)
        
        return CorrectedRetrievalResult(
            original_query=query,
            documents=corrected_docs,
            verification_results=verification_results,
            correction_count=len([d for d in corrected_docs if d.was_corrected])
        )
```

#### Día 19-20: Confidence Scoring & Response Evaluation

```python
# src/application/services/response_evaluator.py
class ResponseEvaluator:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def assess_confidence(self,
                              response: str,
                              context: List[RetrievedDocument]) -> float:
        # 1. Verificar grounding en contexto
        grounding_score = await self._check_grounding(response, context)
        
        # 2. Verificar coherencia interna
        coherence_score = await self._check_coherence(response)
        
        # 3. Detectar posibles alucinaciones
        hallucination_score = await self._detect_hallucinations(
            response, context
        )
        
        # 4. Calcular score final
        confidence = (
            0.4 * grounding_score +
            0.3 * coherence_score +
            0.3 * (1 - hallucination_score)
        )
        
        return confidence
    
    async def _check_grounding(self, 
                              response: str, 
                              context: List[RetrievedDocument]) -> float:
        """Verificar que la respuesta esté basada en el contexto"""
        prompt = f"""
        Response: {response}
        
        Context documents:
        {self._format_context(context)}
        
        Rate how well the response is grounded in the provided context.
        Output a score from 0-1 where:
        1 = Fully grounded, all claims supported
        0 = Not grounded, claims not in context
        """
        
        score = await self.llm_service.evaluate(prompt)
        return float(score)
```

### Semana 17-18: Advanced Features & Frontend

#### Día 21-23: Long RAG Implementation

```python
# src/application/services/long_rag_service.py
class LongRAGService:
    """Manejo de documentos largos sin chunking tradicional"""
    
    def __init__(self,
                 encoder: LongformerEncoder,
                 generator: LLMService):
        self.encoder = encoder
        self.generator = generator
        self.max_length = 4096  # Longformer max
    
    async def process_long_document(self, 
                                  document: str,
                                  query: str) -> LongRAGResult:
        # 1. Encoding del documento completo con Longformer
        doc_encoding = await self.encoder.encode_long_document(
            document, max_length=self.max_length
        )
        
        # 2. Attention-based retrieval
        relevant_segments = await self._extract_relevant_segments(
            doc_encoding, query
        )
        
        # 3. Generación con contexto extendido
        response = await self.generator.generate_with_long_context(
            query=query,
            segments=relevant_segments,
            full_encoding=doc_encoding
        )
        
        return LongRAGResult(
            response=response,
            relevant_segments=relevant_segments,
            attention_scores=doc_encoding.attention_scores
        )
    
    async def _extract_relevant_segments(self,
                                       encoding: LongformerEncoding,
                                       query: str) -> List[TextSegment]:
        """Extraer segmentos relevantes basados en attention scores"""
        query_encoding = await self.encoder.encode_query(query)
        
        # Calcular relevancia usando global attention
        attention_scores = self._compute_attention_scores(
            query_encoding, encoding
        )
        
        # Extraer top-k segmentos
        top_segments = self._extract_top_segments(
            encoding.tokens, attention_scores, k=5
        )
        
        return top_segments
```

#### Día 24-26: Frontend Dashboard (Next.js)

```typescript
// frontend/src/app/search/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { useSocket } from '@/hooks/useSocket';
import { SearchInterface } from '@/components/SearchInterface';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { ConfidenceIndicator } from '@/components/ConfidenceIndicator';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingResponse, setStreamingResponse] = useState('');
  
  const socket = useSocket();
  
  useEffect(() => {
    if (socket) {
      socket.on('retrieval_progress', (data) => {
        // Actualizar UI con progreso de retrieval
        console.log('Retrieval progress:', data);
      });
      
      socket.on('response_chunk', (chunk) => {
        // Streaming de respuesta
        setStreamingResponse(prev => prev + chunk);
      });
    }
  }, [socket]);
  
  const handleSearch = async () => {
    setIsLoading(true);
    setStreamingResponse('');
    
    try {
      const response = await fetch('/api/v1/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query,
          use_self_rag: true,
          stream: true
        })
      });
      
      const data = await response.json();
      setResults(data);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="container mx-auto p-4">
      <SearchInterface
        query={query}
        onQueryChange={setQuery}
        onSearch={handleSearch}
        isLoading={isLoading}
      />
      
      {results && (
        <>
          <ConfidenceIndicator score={results.confidence_score} />
          <ResultsDisplay
            results={results}
            streamingResponse={streamingResponse}
          />
        </>
      )}
    </div>
  );
}
```

```typescript
// frontend/src/components/MultimodalUpload.tsx
import { useDropzone } from 'react-dropzone';
import { useState } from 'react';

export function MultimodalUpload({ onUpload }) {
  const [files, setFiles] = useState([]);
  
  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg'],
      'text/*': ['.txt', '.md']
    },
    onDrop: acceptedFiles => {
      setFiles(acceptedFiles);
      processFiles(acceptedFiles);
    }
  });
  
  const processFiles = async (files) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    const response = await fetch('/api/v1/documents/multimodal', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    onUpload(result);
  };
  
  return (
    <div {...getRootProps()} className="border-2 border-dashed p-8">
      <input {...getInputProps()} />
      <p>Drag & drop files here (PDFs, images, text)</p>
      {files.length > 0 && (
        <div className="mt-4">
          <h3>Files to process:</h3>
          {files.map(file => (
            <div key={file.name}>{file.name}</div>
          ))}
        </div>
      )}
    </div>
  );
}
```

#### Día 27-30: WebSocket Integration & Real-time Features

```python
# src/presentation/websocket/handlers.py
from fastapi import WebSocket
from typing import Dict, Any
import asyncio

class RAGWebSocketHandler:
    def __init__(self, 
                 self_rag_service: SelfRAGOrchestrator,
                 connection_manager: ConnectionManager):
        self.self_rag_service = self_rag_service
        self.connection_manager = connection_manager
    
    async def handle_search_request(self,
                                  websocket: WebSocket,
                                  data: Dict[str, Any]):
        query = Query(text=data["query"])
        
        # Callback para enviar actualizaciones
        async def progress_callback(update: Dict):
            await websocket.send_json({
                "type": "retrieval_progress",
                "data": update
            })
        
        # Ejecutar Self-RAG con callbacks
        self.self_rag_service.set_progress_callback(progress_callback)
        
        # Stream de respuesta
        async for chunk in self.self_rag_service.stream_response(query):
            await websocket.send_json({
                "type": "response_chunk",
                "data": chunk
            })
        
        # Enviar resultado final
        final_result = await self.self_rag_service.get_final_result()
        await websocket.send_json({
            "type": "search_complete",
            "data": final_result.to_dict()
        })

# src/presentation/api/websocket.py
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    handler = RAGWebSocketHandler(self_rag_service, connection_manager)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "search":
                await handler.handle_search_request(websocket, data)
            elif data["type"] == "feedback":
                await handler.handle_feedback(websocket, data)
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
```

## 🔧 Configuración Específica Phase 3

### 1. Multimodal Models Setup

```python
# config/multimodal_config.py
MULTIMODAL_CONFIG = {
    "vision_model": "gpt-4-vision-preview",
    "clip_model": "openai/clip-vit-large-patch14",
    "image_size": (224, 224),
    "max_images_per_document": 50,
    "supported_image_formats": ["jpg", "jpeg", "png", "gif", "bmp"]
}
```

### 2. Self-RAG Configuration

```python
# config/self_rag_config.py
SELF_RAG_CONFIG = {
    "max_iterations": 3,
    "confidence_threshold": 0.7,
    "retrieval_batch_size": 20,
    "reformulation_temperature": 0.8,
    "use_hypothetical_documents": True
}
```

### 3. Frontend Environment

```typescript
// frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_ENABLE_MULTIMODAL=true
```

## 📊 Métricas de Éxito Phase 3

### Funcionalidad
- [ ] Procesamiento de imágenes en PDFs
- [ ] Cross-modal search funcionando
- [ ] Self-RAG con auto-corrección
- [ ] Dashboard interactivo con WebSockets
- [ ] Streaming de respuestas

### Calidad
- [ ] Accuracy de vision-text extraction > 90%
- [ ] Self-RAG confidence calibration > 0.85
- [ ] Reducción de alucinaciones > 50%
- [ ] UI responsiva < 100ms

### Performance
- [ ] Procesamiento multimodal < 2s por imagen
- [ ] Self-RAG iterations < 3 en promedio
- [ ] WebSocket latency < 50ms
- [ ] Frontend load time < 2s

## 🚨 Riesgos y Mitigaciones

### Riesgo 1: Costo de Vision APIs
**Mitigación**: Caching agresivo, usar modelos locales cuando sea posible

### Riesgo 2: Complejidad de Self-RAG
**Mitigación**: Comenzar con lógica simple, añadir complejidad gradualmente

### Riesgo 3: Sincronización frontend-backend
**Mitigación**: Usar queues para manejo de estado, implementar retry logic

## 📝 Checklist de Completación

- [ ] Vision-Language models integrados
- [ ] Multimodal embeddings con CLIP
- [ ] Procesamiento de documentos con imágenes
- [ ] Cross-modal search implementado
- [ ] Self-RAG orchestrator completo
- [ ] Corrective RAG funcionando
- [ ] Confidence scoring calibrado
- [ ] Long RAG para documentos extensos
- [ ] Frontend Next.js configurado
- [ ] Componentes UI principales
- [ ] WebSocket integration
- [ ] Real-time streaming
- [ ] Tests de integración multimodal
- [ ] Documentación de API actualizada