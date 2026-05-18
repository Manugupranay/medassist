#!/bin/bash
# MedAssist AI — Quick Setup Script

echo "🏥 MedAssist AI Setup"
echo "====================="

# Check Python
python3 --version || { echo "❌ Python 3 required"; exit 1; }

# Backend setup
cd backend

# Create .env if not exists
if [ ! -f .env ]; then
  cp .env.example .env
  echo "📝 Created .env — add your ANTHROPIC_API_KEY"
fi

# Install deps
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run:"
echo "  cd backend"
echo "  uvicorn main:app --reload --port 8000"
echo ""
echo "Then open: frontend/index.html in your browser"
echo ""
echo "API docs: http://localhost:8000/docs"
