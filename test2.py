from test1 import *

@decorator
def greet():
    return "Hello, World!"

if __name__ == "__main__":
    import inspect

    def my_function(a: int, b: str = "default", *args, **kwargs):
        pass

    # Get the signature
    sig = inspect.signature(my_function)

    # Loop through parameters to see their details
    for name, param in sig.parameters.items():
        print(f"Name: {name}")
        print(f"  Kind: {param.kind}")          # e.g., POSITIONAL_OR_KEYWORD
        print(f"  Default: {param.default}")    # e.g., 'default' or <class 'inspect._empty'>
        print(f"  Annotation: {param.annotation}") # e.g., <class 'int'>

