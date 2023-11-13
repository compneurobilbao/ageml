import ageml
import sys

def main():
    """Choose between interactive command line or command line 
    based on wether there are no flags when running script"""

    if len(sys.argv) > 1:
        ageml.ui.CLI()
    else:
        ageml.ui.InteractiveCLI()

if __name__ == '__main__':
    main()