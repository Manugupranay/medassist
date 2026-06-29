# Contributing to MedAssist AI

Thank you for your interest in contributing to MedAssist! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Bugs

1. Check the [existing issues](../../issues) to avoid duplicates.
2. 2. Open a new issue with a clear title and description.
   3. 3. Include steps to reproduce, expected behavior, and actual behavior.
      4. 4. Add relevant logs or screenshots.
        
         5. ### Suggesting Features
        
         6. 1. Open an issue with the `enhancement` label.
            2. 2. Describe the feature and its use case clearly.
               3. 3. Discuss the implementation approach if possible.
                 
                  4. ### Submitting Pull Requests
                 
                  5. 1. Fork the repository and create a new branch from `main`:
                     2.    ```bash
                              git checkout -b feat/your-feature-name
                              ```

                           2. Set up your development environment:
                           3.    ```bash
                                    cd backend
                                    pip install -r requirements.txt
                                    pip install -r requirements-dev.txt  # if available
                                    ```

                                 3. Make your changes following the coding standards below.
                             
                                 4. 4. Write or update tests for your changes.
                                   
                                    5. 5. Run the test suite:
                                       6.    ```bash
                                                pytest tests/ -v
                                                ```

                                             6. Commit with a descriptive message following [Conventional Commits](https://www.conventionalcommits.org/):
                                             7.    ```
                                                      feat: add drug interaction checker endpoint
                                                      fix: resolve null pointer in RAG pipeline
                                                      docs: update API documentation
                                                      ```

                                                   7. Push and open a Pull Request against `main`.
                                               
                                                   8. ## Coding Standards
                                               
                                                   9. ### Python (Backend)
                                               
                                                   10. - Follow [PEP 8](https://pep8.org/) style guide.
                                                       - - Use type hints for all function signatures.
                                                         - - Write docstrings for all public functions and classes.
                                                           - - Maximum line length: 127 characters.
                                                             - - Use `black` for formatting and `flake8` for linting.
                                                              
                                                               - ### Commit Messages
                                                              
                                                               - Follow the Conventional Commits specification:
                                                               - - `feat:` — New feature
                                                                 - - `fix:` — Bug fix
                                                                   - - `docs:` — Documentation changes
                                                                     - - `test:` — Adding or updating tests
                                                                       - - `refactor:` — Code refactoring
                                                                         - - `chore:` — Maintenance tasks
                                                                          
                                                                           - ## Development Setup
                                                                          
                                                                           - ```bash
                                                                             # Clone the repo
                                                                             git clone https://github.com/Manugupranay/medassist.git
                                                                             cd medassist

                                                                             # Backend setup
                                                                             cd backend
                                                                             python -m venv venv
                                                                             source venv/bin/activate  # On Windows: venv\Scripts\activate
                                                                             pip install -r requirements.txt

                                                                             # Set up environment variables
                                                                             cp .env.example .env
                                                                             # Edit .env with your API keys

                                                                             # Run the backend
                                                                             uvicorn main:app --reload
                                                                             ```

                                                                             ## Project Structure

                                                                             ```
                                                                             medassist/
                                                                             ├── backend/          # FastAPI + LangGraph backend
                                                                             │   ├── agents/       # AI agent definitions
                                                                             │   ├── api/          # REST API endpoints
                                                                             │   ├── models/       # Pydantic models
                                                                             │   ├── services/     # Business logic
                                                                             │   └── tests/        # Unit and integration tests
                                                                             └── frontend/         # React frontend
                                                                             ```

                                                                             ## Questions?

                                                                             Feel free to open an issue or start a discussion. We appreciate all contributions!
                                                                             
