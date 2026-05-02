#import sys
import logging
from neural_network import NeuralNetwork

logger = logging.getLogger(__name__)

def main():
    #argv = sys.argv
    logging.basicConfig(filename='myapp.log', level=logging.DEBUG)

if __name__ == "__main__":
    main()
