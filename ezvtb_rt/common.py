from abc import ABC, abstractmethod
import numpy as np
from typing import List
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
DEFAULT_EZVTB_DATA = os.path.join(current_dir, '..','data')
EZVTB_DATA = os.getenv('EZVTB_DATA', DEFAULT_EZVTB_DATA)

class Core(ABC):
    @abstractmethod
    def setImage(self, img:np.ndarray):
        pass  # Call at initialization

    @abstractmethod
    def inference(self, pose:np.ndarray) -> List[np.ndarray]:
        pass  # sync to get ONE OF the results 
    