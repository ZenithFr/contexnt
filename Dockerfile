FROM python:3.10-slim

WORKDIR /app

# Install git for the auto-commit functionality
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Run the FastMCP server
CMD ["python", "-m", "src.server"]
