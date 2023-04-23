from termcolor import colored


def print_heading(heading: str, color: str):
    return print(colored(f"\n*****{heading}*****", color))
