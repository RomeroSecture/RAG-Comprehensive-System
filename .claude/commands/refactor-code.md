---
allowed-tools: ["Read", "Edit", "MultiEdit", "Write", "Grep", "TodoWrite", "Bash"]
description: "Refactoriza código para mejorar calidad, mantener Clean Architecture y aplicar mejores prácticas"
---

Refactoriza el código en: $ARGUMENTS

## 🔧 Refactorización Inteligente de Código

### 1. **Análisis del Código Actual**

#### Identificar Code Smells
```python
code_smells_to_check = {
    "Long Methods": "funciones > 50 líneas",
    "Large Classes": "clases > 300 líneas",
    "Duplicate Code": "código repetido",
    "Feature Envy": "métodos que usan más otra clase",
    "Data Clumps": "grupos de datos que viajan juntos",
    "Primitive Obsession": "uso excesivo de tipos primitivos",
    "Long Parameter Lists": "más de 4 parámetros",
    "Divergent Change": "clase que cambia por múltiples razones",
    "Shotgun Surgery": "cambio requiere modificar muchos lugares",
    "Lazy Class": "clases que hacen muy poco"
}
```

#### Verificar Principios SOLID
- **S**ingle Responsibility
- **O**pen/Closed
- **L**iskov Substitution
- **I**nterface Segregation
- **D**ependency Inversion

#### Validar Clean Architecture
- Dependencias correctas entre capas
- No hay lógica de negocio en controllers
- Entidades sin dependencias externas
- Use cases orquestando correctamente

### 2. **Plan de Refactorización**

Crear plan detallado con TodoWrite:
1. Identificar componentes a refactorizar
2. Orden de refactorización (menos a más dependencias)
3. Tests necesarios antes de refactorizar
4. Verificaciones post-refactorización

### 3. **Técnicas de Refactorización**

#### Extract Method
```python
# Antes
def process_order(order):
    # Validar orden - 20 líneas
    if not order.items:
        raise ValueError("Order empty")
    # ... más validaciones
    
    # Calcular total - 15 líneas
    total = 0
    for item in order.items:
        total += item.price * item.quantity
    # ... más cálculos
    
    # Aplicar descuentos - 10 líneas
    if order.customer.is_premium:
        total *= 0.9
    # ... más lógica

# Después
def process_order(order):
    validate_order(order)
    total = calculate_total(order)
    total = apply_discounts(total, order)
    return total

def validate_order(order):
    if not order.items:
        raise ValueError("Order empty")
    # ... validaciones

def calculate_total(order):
    return sum(item.price * item.quantity for item in order.items)

def apply_discounts(total, order):
    if order.customer.is_premium:
        total *= 0.9
    return total
```

#### Extract Class
```python
# Antes - Clase con múltiples responsabilidades
class UserService:
    def create_user(self, data): ...
    def authenticate(self, credentials): ...
    def send_email(self, user, subject, body): ...
    def generate_token(self, user): ...
    def validate_password(self, password): ...

# Después - Responsabilidades separadas
class UserService:
    def __init__(self, auth_service, email_service):
        self.auth_service = auth_service
        self.email_service = email_service
    
    def create_user(self, data): ...

class AuthenticationService:
    def authenticate(self, credentials): ...
    def generate_token(self, user): ...
    def validate_password(self, password): ...

class EmailService:
    def send_email(self, user, subject, body): ...
```

#### Replace Primitive with Value Object
```python
# Antes
class Document:
    def __init__(self, content: str, score: float):
        self.content = content
        self.score = score  # Primitive

# Después
@dataclass(frozen=True)
class SimilarityScore:
    value: float
    
    def __post_init__(self):
        if not 0 <= self.value <= 1:
            raise ValueError("Score must be between 0 and 1")
    
    def is_high_confidence(self) -> bool:
        return self.value > 0.8

class Document:
    def __init__(self, content: str, score: SimilarityScore):
        self.content = content
        self.score = score  # Value Object
```

#### Dependency Injection
```python
# Antes - Dependencias hardcoded
class QueryProcessor:
    def __init__(self):
        self.embedder = OpenAIEmbedder()  # Hardcoded
        self.store = PgVectorStore()      # Hardcoded

# Después - Inyección de dependencias
class QueryProcessor:
    def __init__(self, embedder: EmbedderProtocol, store: VectorStoreProtocol):
        self.embedder = embedder
        self.store = store
```

### 4. **Refactorización por Capa**

#### Domain Layer
- Extraer lógica de negocio de otras capas
- Crear Value Objects para conceptos del dominio
- Implementar métodos de dominio en entidades
- Eliminar dependencias externas

#### Application Layer
- Simplificar use cases largos
- Extraer lógica común a servicios
- Mejorar manejo de errores
- Implementar patrones como Chain of Responsibility

#### Infrastructure Layer
- Aplicar patrón Adapter consistentemente
- Unificar configuración
- Mejorar gestión de conexiones
- Implementar circuit breakers

#### Presentation Layer
- Simplificar controllers
- Extraer validación a middleware
- Unificar respuestas de error
- Mejorar documentación de API

### 5. **Verificación Post-Refactorización**

```bash
# Ejecutar tests
!pytest tests/ -v

# Verificar coverage no ha bajado
!pytest tests/ --cov=src --cov-report=term

# Validar arquitectura
!claude code /validate-architecture

# Verificar performance
!claude code /benchmark-rag-performance
```

### 6. **Documentar Cambios**

Crear archivo de cambios:
```markdown
# Refactorización: [Componente]
**Fecha**: [Timestamp]
**Razón**: [Por qué se refactorizó]

## Cambios Realizados
1. [Cambio 1 - técnica usada]
2. [Cambio 2 - técnica usada]

## Mejoras Obtenidas
- Complejidad reducida de X a Y
- Mejor separación de responsabilidades
- Mayor testabilidad
- [Otras mejoras]

## Riesgos Mitigados
- [Code smell eliminado]
- [Principio SOLID aplicado]
```

### 7. **Patrones de Refactorización RAG**

#### Extraer Estrategias de Retrieval
```python
# Antes - Switch largo
def retrieve(query, strategy):
    if strategy == "vector":
        # 20 líneas de código
    elif strategy == "hybrid":
        # 30 líneas de código
    elif strategy == "graph":
        # 40 líneas de código

# Después - Strategy Pattern
class RetrievalStrategy(Protocol):
    async def retrieve(self, query: Query) -> List[Document]: ...

class VectorRetrieval(RetrievalStrategy):
    async def retrieve(self, query: Query) -> List[Document]: ...

class HybridRetrieval(RetrievalStrategy):
    async def retrieve(self, query: Query) -> List[Document]: ...

class RetrievalOrchestrator:
    def __init__(self, strategies: Dict[str, RetrievalStrategy]):
        self.strategies = strategies
    
    async def retrieve(self, query: Query, strategy: str) -> List[Document]:
        return await self.strategies[strategy].retrieve(query)
```

### 8. **Métricas de Calidad**

Medir antes y después:
- Complejidad ciclomática
- Acoplamiento entre módulos
- Cohesión de clases
- Líneas por método/clase
- Duplicación de código

### 9. **Rollback Plan**

Si algo sale mal:
1. Git stash cambios actuales
2. Revertir al commit anterior
3. Analizar qué falló
4. Re-aplicar refactorización por partes

### 10. **Integración Continua**

Actualizar CI/CD si es necesario:
- Nuevos paths de archivos
- Nuevas dependencias
- Cambios en configuración

## 🎯 Resultado Esperado

Código refactorizado que:
- Es más legible y mantenible
- Sigue principios SOLID
- Respeta Clean Architecture
- Tiene mejor testabilidad
- Mantiene o mejora el performance
- Está bien documentado