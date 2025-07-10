---
allowed-tools: ["Write", "Edit", "Read", "Bash", "TodoWrite"]
description: "Implementa una entidad de dominio siguiendo DDD y Clean Architecture"
---

Implementa una nueva entidad de dominio para el sistema RAG: $ARGUMENTS

## 🏗️ Implementación de Entidad de Dominio

### 1. **Análisis de la Entidad**
Primero, determina:
- Nombre de la entidad
- Propiedades y sus tipos
- Invariantes de negocio
- Comportamientos (métodos)
- Relaciones con otras entidades

### 2. **Crear la Entidad**

#### Estructura Base
```python
# src/domain/entities/{entity_name}.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from src.domain.value_objects import {relevant_value_objects}
from src.domain.exceptions import DomainException


@dataclass
class {EntityName}:
    """
    {Descripción de la entidad y su propósito en el dominio}
    
    Invariantes:
    - {Lista de reglas de negocio que siempre deben cumplirse}
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    
    # Properties
    {property_name}: {type}
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate domain invariants.
        
        Raises:
            DomainException: If any invariant is violated
        """
        # Implement validation logic
        pass
    
    # Business Methods
    def {business_method}(self, ...) -> {ReturnType}:
        """
        {Descripción del comportamiento}
        
        Args:
            {args}
            
        Returns:
            {return description}
            
        Raises:
            DomainException: {when}
        """
        # Implement business logic
        pass
```

### 3. **Crear Value Objects Relacionados**

```python
# src/domain/value_objects/{value_object_name}.py
from dataclasses import dataclass
from typing import Any

from src.domain.exceptions import ValidationException


@dataclass(frozen=True)
class {ValueObjectName}:
    """
    {Descripción del value object}
    """
    value: {type}
    
    def __post_init__(self) -> None:
        self._validate()
    
    def _validate(self) -> None:
        """Validate the value object."""
        # Validation logic
        if not self._is_valid():
            raise ValidationException(f"Invalid {self.__class__.__name__}: {self.value}")
    
    def _is_valid(self) -> bool:
        """Check if the value is valid."""
        # Implementation
        return True
    
    def __str__(self) -> str:
        return str(self.value)
```

### 4. **Definir Excepciones de Dominio**

```python
# src/domain/exceptions.py (update if exists)
class {EntityName}Exception(DomainException):
    """Base exception for {EntityName} domain errors."""
    pass

class {SpecificBusinessRuleViolation}(EntityNameException):
    """Raised when {specific rule} is violated."""
    pass
```

### 5. **Crear Repository Interface**

```python
# src/domain/repositories/{entity_name}_repository.py
from abc import ABC, abstractmethod
from typing import Optional, List
from uuid import UUID

from src.domain.entities.{entity_name} import {EntityName}


class {EntityName}Repository(ABC):
    """
    Abstract repository for {EntityName} persistence.
    """
    
    @abstractmethod
    async def save(self, entity: {EntityName}) -> {EntityName}:
        """Persist the entity."""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Optional[{EntityName}]:
        """Retrieve entity by ID."""
        pass
    
    @abstractmethod
    async def find_by_{criteria}(self, {criteria}: {Type}) -> List[{EntityName}]:
        """Find entities by specific criteria."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> None:
        """Delete the entity."""
        pass
    
    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists."""
        pass
```

### 6. **Crear Tests Unitarios**

```python
# tests/unit/domain/entities/test_{entity_name}.py
import pytest
from uuid import uuid4
from datetime import datetime

from src.domain.entities.{entity_name} import {EntityName}
from src.domain.exceptions import DomainException


class Test{EntityName}:
    """Unit tests for {EntityName} entity."""
    
    def test_create_valid_{entity_name}(self):
        """Test creating a valid {entity_name}."""
        # Arrange
        {setup_test_data}
        
        # Act
        entity = {EntityName}(
            {properties}
        )
        
        # Assert
        assert entity.id is not None
        assert entity.{property} == {expected_value}
    
    def test_invalid_{property}_raises_exception(self):
        """Test that invalid {property} raises exception."""
        # Arrange & Act & Assert
        with pytest.raises(DomainException):
            {EntityName}({invalid_data})
    
    def test_{business_method}_behavior(self):
        """Test {business_method} behavior."""
        # Arrange
        entity = {create_valid_entity}
        
        # Act
        result = entity.{business_method}({args})
        
        # Assert
        assert result == {expected_result}
    
    def test_invariant_{invariant_name}(self):
        """Test that {invariant_name} is enforced."""
        # Test specific business rule
        pass
```

### 7. **Crear Factory si es Complejo**

```python
# src/domain/factories/{entity_name}_factory.py
from typing import Dict, Any
from uuid import UUID

from src.domain.entities.{entity_name} import {EntityName}
from src.domain.value_objects import {ValueObjects}


class {EntityName}Factory:
    """Factory for creating {EntityName} instances."""
    
    @staticmethod
    def create_from_dict(data: Dict[str, Any]) -> {EntityName}:
        """
        Create entity from dictionary data.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            {EntityName} instance
        """
        # Parse and validate data
        # Create value objects
        # Return entity
        pass
    
    @staticmethod
    def create_for_{use_case}({params}) -> {EntityName}:
        """Create entity for specific use case."""
        # Specialized creation logic
        pass
```

### 8. **Documentar en Domain Model**

Actualizar documentación:
```markdown
# src/domain/README.md

## Entities

### {EntityName}
- **Purpose**: {What it represents}
- **Invariants**: 
  - {List business rules}
- **Key Behaviors**:
  - {List main methods}
- **Relationships**:
  - {How it relates to other entities}
```

### 9. **Integración con el Sistema**

#### Registrar en __init__.py
```python
# src/domain/entities/__init__.py
from .{entity_name} import {EntityName}

__all__ = ["{EntityName}", ...]
```

#### Ejemplo de Uso
```python
# Example usage in use case
from src.domain.entities.{entity_name} import {EntityName}

class Create{EntityName}UseCase:
    async def execute(self, data: CreateDTO) -> {EntityName}:
        entity = {EntityName}(
            # Map DTO to entity
        )
        return await self.repository.save(entity)
```

### 10. **Verificación Final**

Checklist:
- [ ] La entidad no tiene dependencias externas al dominio
- [ ] Todos los invariantes están validados
- [ ] Los métodos representan comportamientos del negocio
- [ ] Tests unitarios cubren todos los casos
- [ ] No hay lógica de persistencia o presentación
- [ ] La documentación está completa
- [ ] Sigue las convenciones del proyecto

## 🎯 Resultado Esperado

Una entidad de dominio pura que:
- Encapsula reglas de negocio
- Es totalmente testeable
- No depende de frameworks
- Es fácil de entender y mantener