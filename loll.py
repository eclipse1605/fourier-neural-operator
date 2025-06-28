                      

import os
import sys
import io
import tokenize

                                     
SKIP_DIRS = {'.venv', '__pycache__'}


def remove_comments_from_file(path: str) -> None:
    """
    Remove all comments from a Python file at the given path.
    The file is read, comments are filtered out, and the cleaned code is written back.
    """
    try:
                                  
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()

                                          
        tokens = tokenize.generate_tokens(io.StringIO(src).readline)
        filtered_tokens = [tok for tok in tokens if tok.type != tokenize.COMMENT]
        cleaned = tokenize.untokenize(filtered_tokens)

                                                 
        with open(path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"Processed: {path}")

    except Exception as e:
        print(f"Error processing {path}: {e}")


def main(root: str) -> None:
    """
    Walk through the directory tree starting at root,
    skipping specified directories, and remove comments from all .py files.
    """
    for dirpath, dirnames, filenames in os.walk(root):
                                                               
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            if fname.endswith('.py'):
                full_path = os.path.join(dirpath, fname)
                remove_comments_from_file(full_path)


if __name__ == '__main__':
                                                 
    root_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(root_dir)
