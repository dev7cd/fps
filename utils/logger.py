"""
utils/logger.py
Configuration centralisée du logging
"""

import logging
import sys
from typing import Optional

def setup_logging(level=logging.INFO, log_file: Optional[str] = None):
    """Configure le logging pour l'application."""
    
    # Formateur personnalisé
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Appliquer le formateur au StreamHandler
    handlers[0].setFormatter(formatter)
    
    # Configuration principale
    # On force la config pour écraser les configs par défaut potentielles
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True 
    )
    
    # Réduire le verbosity de certaines librairies tierces bruyantes
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

class LoggingMixin:
    """Mixin pour ajouter facilement le logging aux classes."""
    
    @property
    def logger(self):
        name = '.'.join([__name__, self.__class__.__name__])
        return logging.getLogger(name)