# Conversation Filter

![CI](https://github.com/jamescranston/convo-filter/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/jamescranston/convo-filter/branch/main/graph/badge.svg)](https://codecov.io/gh/jamescranston/convo-filter)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A Python tool that processes PDF conversation history and filters it by specified topics using Anthropic's Claude AI model. This tool is particularly useful for analyzing large conversation histories and extracting relevant discussions based on topics of interest.

## Author

James Cranston

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Features

- Extract text from PDF documents or compressed PDF archives (.pdf.zip)
- Filter conversations by multiple topics using Claude AI
- Load topics from a text file (one topic per line)
- Preserve conversation context when filtering
- **Privacy-focused**:
  - Accepts compressed PDF input (.pdf.zip)
  - No temporary files with PDF content
  - No sensitive content printed to terminal
  - Output automatically compressed (PDF or text)
- Secure API key management
- **Cost optimization**: Batch multiple topic checks in a single API call
- **Performance optimization**: Process pages in parallel with multithreading
- **Topic continuity**: Group consecutive pages discussing the same topic

## Installation

### Prerequisites

- Python 3.10 or higher
- Anthropic API key

### Using Poetry (Recommended)

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/convo-filter.git
   cd convo-filter
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Using pip

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/convo-filter.git
   cd convo-filter
   ```

2. Install the package:
   ```bash
   pip install .
   ```

## API Key Setup

There are several ways to provide your Anthropic API key:

### Option 1: Environment Variable (Recommended)

Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Option 2: Command-line Argument

Provide your API key directly when running the program:

```bash
convo-filter path/to/your/conversation.pdf.zip --api-key "your-api-key-here"
```

## Usage

### Basic Usage

```bash
convo-filter path/to/your/conversation.pdf.zip
```

This will filter the conversation using default topics and output the results to a compressed PDF file.

### Command Line Options

```bash
# Specify topics file
convo-filter path/to/your/conversation.pdf.zip --topics-file topics.txt

# Specify topics directly
convo-filter path/to/your/conversation.pdf.zip --topics "vacation" "family" "work"

# Custom output file (defaults to compressed PDF)
convo-filter path/to/your/conversation.pdf.zip --output filtered_results.pdf.zip

# Output as compressed text file
convo-filter path/to/your/conversation.pdf.zip --text-output

# Specify Claude model
convo-filter path/to/your/conversation.pdf.zip --model "claude-3-opus-20240229"

# Performance settings
convo-filter path/to/your/conversation.pdf.zip --batch-size 30 --pages-per-batch 10 --workers 8

# Adjust matching strictness
convo-filter path/to/your/conversation.pdf.zip --strictness low|medium|high

# Show detailed output
convo-filter path/to/your/conversation.pdf.zip --no-privacy
```

### Comprehensive Example

Here's an example that combines multiple options for a complex use case:

```bash
convo-filter input_convo.pdf.zip \
  --topics-file my_topics.txt \
  --output filtered_results.pdf.zip \
  --model "claude-3-opus-20240229" \
  --batch-size 30 \
  --pages-per-batch 10 \
  --workers 8 \
  --strictness high \
  --no-group \
  --no-privacy
```

This command:
- Processes a compressed PDF input file
- Uses topics from a custom file
- Outputs to a compressed PDF
- Uses the Claude 3 Opus model
- Processes 30 topics per API call
- Handles 10 pages per batch
- Uses 8 worker threads
- Uses high strictness for topic matching
- Disables grouping of consecutive pages
- Shows detailed output (disables privacy mode)

### Topics File Format

Create a text file with one topic per line. Lines starting with # are treated as comments:

```text
# Example neutral topics
meeting notes
project updates
action items
work
career change
```

## Input and Output File Locations

- You can specify the full or relative path to your input PDF (or ZIP) and output file.
- There are no required input or output directories; files can be anywhere on your system.
- Example usage:
  ```bash
  convo-filter /Users/yourname/Documents/convo.pdf.zip --output /Users/yourname/Desktop/filtered_results.pdf.zip
  ```
- If you do not specify an output path, the result will be saved in the current working directory with a default name.

## Performance and Cost Optimization

The tool includes several optimizations:

1. **Batched API Calls**: Multiple topics and pages are processed in a single API call
2. **Multithreading**: Parallel processing of page batches
   - Default: 2 workers
   - Maximum: (CPU cores - 1)
   - ⚠️ **Warning**: Setting too many workers can actually slow down processing due to:
     - API rate limiting (45 requests per minute)
     - Context switching overhead
     - Memory usage
     - Recommended maximum: 4-8 workers for most systems
3. **Built-in Rate Limiting**:
   - Automatically enforces Anthropic's API rate limits
   - Uses thread-safe implementation with Python's `threading` module
   - Maintains a rolling window of API calls
   - Automatically pauses when approaching rate limits
   - Provides real-time feedback on rate limit waits
4. **Progress Tracking**: Real-time progress and time estimates
5. **Topic Continuity**: Groups consecutive pages discussing the same topic
6. **API Call Counting**: Tracks and reports API usage

## Privacy Features

- Compressed input/output files
- Automatic temporary file cleanup
- Local API key storage
- Privacy mode (enabled by default)
- Comment filtering in topics file

## Development

### Setting Up Development Environment

1. Clone the repository and install development dependencies:
   ```bash
   git clone https://github.com/yourusername/convo-filter.git
   cd convo-filter
   poetry install
   ```

2. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

### Running Tests

```bash
poetry run pytest
```

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Static type checking
- **ruff**: Linting
- **pytest**: Testing
- **pre-commit**: Git hooks

### Project Structure

```
convo_filter/
├── cli/            # Command-line interface
├── core/           # Core functionality
├── utils/          # Utility functions
└── tests/          # Test suite
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## License

MIT License

Copyright (c) 2025 James Cranston

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Support

For issues and feature requests, please use the GitHub issue tracker.
