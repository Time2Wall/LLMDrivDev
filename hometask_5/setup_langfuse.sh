#!/bin/bash
# Setup script for local Langfuse instance

echo "=========================================="
echo "Langfuse Local Setup Script"
echo "=========================================="

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "Docker found. Starting Langfuse with Docker..."
    docker-compose -f docker-compose.langfuse.yml up -d

    echo ""
    echo "Waiting for services to start..."
    sleep 10

    echo ""
    echo "Langfuse is starting at http://localhost:3000"
    echo "First time setup:"
    echo "  1. Go to http://localhost:3000"
    echo "  2. Create an account"
    echo "  3. Create a project"
    echo "  4. Go to Settings > API Keys"
    echo "  5. Copy Public Key and Secret Key"
    echo "  6. Update .env file with your keys"
else
    echo "Docker not found. Please use one of these options:"
    echo ""
    echo "Option 1: Install Docker and run:"
    echo "  docker-compose -f docker-compose.langfuse.yml up -d"
    echo ""
    echo "Option 2: Use Langfuse Cloud (Free tier):"
    echo "  1. Go to https://cloud.langfuse.com"
    echo "  2. Sign up for free account"
    echo "  3. Create a project"
    echo "  4. Get API keys from Settings > API Keys"
    echo "  5. Update .env with:"
    echo "     LANGFUSE_HOST=https://cloud.langfuse.com"
    echo "     LANGFUSE_PUBLIC_KEY=pk-lf-..."
    echo "     LANGFUSE_SECRET_KEY=sk-lf-..."
fi

echo ""
echo "=========================================="
echo "Setup Complete"
echo "=========================================="
