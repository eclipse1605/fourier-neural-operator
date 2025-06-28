#!/usr/bin/env python3
"""
Docstring Remover (loll2.py)

This script removes multiline docstrings from Python files.
Unlike loll.py that removes single-line comments, this focuses on 
multiline docstrings (triple-quoted strings used for documentation).
"""

import os
import sys
import re
import ast
from typing import List, Tuple

# Directories to skip during traversal
SKIP_DIRS = {'.venv', '__pycache__', 'build', 'dist', '.git', '.idea', '.vscode'}

class DocstringRemover(ast.NodeVisitor):
    """AST visitor that identifies docstring positions in the code."""
    
    def __init__(self):
        self.docstring_ranges = []
    
    def visit_Module(self, node):
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            self.docstring_ranges.append((node.body[0].lineno, node.body[0].end_lineno))
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            self.docstring_ranges.append((node.body[0].lineno, node.body[0].end_lineno))
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            self.docstring_ranges.append((node.body[0].lineno, node.body[0].end_lineno))
        self.generic_visit(node)


def remove_docstrings(file_path: str) -> bool:
    """
    Remove all docstrings from a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse the file and find docstrings
        tree = ast.parse(content)
        visitor = DocstringRemover()
        visitor.visit(tree)
        
        if not visitor.docstring_ranges:
            # No docstrings found
            return True
            
        # Convert to lines for easier manipulation
        lines = content.split('\n')
        new_lines = []
        
        # Remove docstrings
        current_line = 1
        for lineno, end_lineno in sorted(visitor.docstring_ranges):
            # Keep lines before the docstring
            new_lines.extend(lines[current_line-1:lineno-1])
            current_line = end_lineno + 1
        
        # Add remaining lines
        if current_line <= len(lines):
            new_lines.extend(lines[current_line-1:])
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
            
        return True
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def remove_docstrings_regex(file_path: str) -> bool:
    """
    Remove docstrings using regex pattern matching.
    This is a fallback method when AST parsing fails.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern for multiline docstrings (triple quotes with optional whitespace)
        pattern = r'(?m)(^\s*["\'])(["\'])\1(?:\s*$)(.*?)(?:\n[ \t]*)(?:\1\2\1)(?:\s*$)'
        
        # Remove docstrings
        cleaned_content = re.sub(pattern, '', content, flags=re.DOTALL)
        
        # Also remove module, class and function level docstrings
        cleaned_content = re.sub(r'(?m)(def|class)\s+\w+\s*\([^)]*\)\s*:\s*\n\s*(["\'])\2\2[\s\S]*?\2\2\2', 
                                r'\1 \2():', cleaned_content)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
            
        return True
    
    except Exception as e:
        print(f"Error processing {file_path} with regex: {e}")
        return False


def process_file(file_path: str) -> None:
    """Process a single Python file to remove docstrings."""
    print(f"Processing: {file_path}")
    
    # Try AST-based removal first, fall back to regex if it fails
    if not remove_docstrings(file_path):
        remove_docstrings_regex(file_path)


def main(root_dir: str) -> None:
    """
    Walk through directory tree and remove docstrings from all Python files.
    
    Args:
        root_dir: Root directory to start traversal
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip specified directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                process_file(file_path)


if __name__ == '__main__':
    # Use command line arg as root dir, or current dir if not specified
    root_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    print(f"Removing docstrings from Python files in {root_dir}")
    main(root_dir)
    print("Done!")
