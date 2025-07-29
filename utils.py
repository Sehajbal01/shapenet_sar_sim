import os

def get_next_path(path):
    """
    Get the next available path by appending a number to the base path.
    If the path already exists, it will increment the number until it finds an available one.
    """
    split_path = path.split('.')
    assert len(split_path) > 1, "Path must have an extension to append a number."
    base_path = '.'.join(split_path[:-1])
    extension = split_path[-1]
    
    i = 0
    while True:
        new_path = '%s_%02d.%s' % (base_path, i, extension)
        if not os.path.exists(new_path):
            return new_path
        i += 1