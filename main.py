import ageml
import sys

def main():
    """Choose between interactive command line or command line 
    based on wether there are no flags when running script"""

    if len(sys.argv) > 1:
        ageml.ui.CLI().run()
    else:
        ageml.ui.InteractiveCLI().command_interface()

if __name__ == '__main__':
    main()