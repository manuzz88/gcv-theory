#!/bin/bash
# Script di avvio rapido per test GCV

echo "=============================================="
echo "🚀 TEST DEFINITIVI TEORIA GCV"
echo "=============================================="
echo ""
echo "Questo script eseguirà l'analisi completa per determinare"
echo "se la GCV è competitiva con la materia oscura."
echo ""
echo "Test inclusi:"
echo "  ✓ Test 2: Weak Lensing (SDSS)"
echo "  ✓ Test 3: Cluster Merger (Bullet/El Gordo)"
echo ""
echo "Tempo stimato: 30-60 minuti"
echo "Hardware: Ottimizzato per GPU (funziona anche su CPU)"
echo ""
echo "=============================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trovato!"
    echo "   Installa Python 3.8+ prima di continuare"
    exit 1
fi

echo "✓ Python 3 trovato: $(python3 --version)"
echo ""

# Check dipendenze principali
echo "Controllo dipendenze critiche..."

MISSING=""

if ! python3 -c "import numpy" 2>/dev/null; then
    MISSING="$MISSING numpy"
fi

if ! python3 -c "import scipy" 2>/dev/null; then
    MISSING="$MISSING scipy"
fi

if ! python3 -c "import matplotlib" 2>/dev/null; then
    MISSING="$MISSING matplotlib"
fi

if ! python3 -c "import astropy" 2>/dev/null; then
    MISSING="$MISSING astropy"
fi

if [ ! -z "$MISSING" ]; then
    echo ""
    echo "⚠️  Dipendenze mancanti:$MISSING"
    echo ""
    echo "Vuoi installarle ora? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo ""
        echo "📦 Installazione dipendenze..."
        python3 -m pip install -r requirements.txt
        
        if [ $? -eq 0 ]; then
            echo "✅ Dipendenze installate!"
        else
            echo "❌ Errore installazione"
            echo "   Prova manualmente: pip install -r requirements.txt"
            exit 1
        fi
    else
        echo "❌ Installazione annullata"
        echo "   Installa manualmente: pip install -r requirements.txt"
        exit 1
    fi
else
    echo "✅ Tutte le dipendenze critiche presenti"
fi

echo ""
echo "=============================================="
echo "Pronto per iniziare!"
echo "=============================================="
echo ""
echo "Opzioni:"
echo ""
echo "  1. Esecuzione COMPLETA automatica (raccomandato)"
echo "     → Setup + Download + Test 2 + Test 3 + Verdetto"
echo ""
echo "  2. Solo ISTRUZIONI (leggi prima di eseguire)"
echo "     → Apre file ISTRUZIONI.md"
echo ""
echo "  3. Esci"
echo ""
echo -n "Scelta (1/2/3): "
read -r choice

case $choice in
    1)
        echo ""
        echo "▶️  Avvio analisi completa..."
        echo ""
        python3 run_all_tests.py
        exit_code=$?
        
        echo ""
        echo "=============================================="
        if [ $exit_code -eq 0 ]; then
            echo "🎉 GCV È COMPETITIVA!"
            echo ""
            echo "Vedi risultati in:"
            echo "  • results/FINAL_VERDICT.md"
            echo "  • plots/*.png"
        elif [ $exit_code -eq 2 ]; then
            echo "⚠️  GCV È PLAUSIBILE (risultati ambigui)"
            echo ""
            echo "Vedi risultati in:"
            echo "  • results/FINAL_VERDICT.md"
        else
            echo "❌ GCV NON È COMPETITIVA"
            echo ""
            echo "Vedi dettagli in:"
            echo "  • results/FINAL_VERDICT.md"
        fi
        echo "=============================================="
        ;;
    
    2)
        echo ""
        echo "📖 Apertura istruzioni..."
        echo ""
        
        if command -v xdg-open &> /dev/null; then
            xdg-open ISTRUZIONI.md
        elif command -v less &> /dev/null; then
            less ISTRUZIONI.md
        else
            cat ISTRUZIONI.md
        fi
        
        echo ""
        echo "Quando sei pronto, esegui:"
        echo "  python3 run_all_tests.py"
        ;;
    
    3)
        echo "Uscita."
        exit 0
        ;;
    
    *)
        echo "Scelta non valida"
        exit 1
        ;;
esac
