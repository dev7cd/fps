"""
 @file logger.py
 @brief Configuration centralisée du logging pour l'application.
 @details Fournit des outils pour initialiser le logging et un Mixin pour les classes.
"""

import logging
import sys
from typing import Optional

def setup_logging(level=logging.INFO, log_file: Optional[str] = None):
    """
    @brief Configure le logging global pour l'application.
    
    Initialise les handlers pour la sortie standard et optionnellement pour un fichier.
    Définit également le format des messages et ajuste le niveau des bibliothèques tierces.
    
    @param level Le niveau de sévérité du logging (ex: logging.INFO, logging.DEBUG).
    @param log_file Chemin optionnel vers un fichier de log.
    """
    
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
    """
    @class LoggingMixin
    @brief Mixin pour ajouter facilement une instance de logger aux classes.
    """
    
    @property
    def logger(self):
        """
        @brief Récupère ou crée un logger spécifique à la classe.
        
        @return Une instance de logging.Logger nommée d'après la classe.
        """
        name = '.'.join([__name__, self.__class__.__name__])
        return logging.getLogger(name)