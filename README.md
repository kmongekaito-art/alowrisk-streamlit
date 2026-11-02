# A Low-Risk ETF Trading Strategy — Streamlit

Réplica en Python del sistema **RS/RSI long-only** de Markos Katsanos descrito en
*A Low-Risk ETF Trading Strategy* y del código AFL de AmiBroker.

## Instalación y ejecución

```powershell
cd C:\Users\koldo\PROYECTO-OPENCODE\ALOWRISK
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`.

## Qué incluye

- **Selector de portfolio** con las dos listas del script original:
  - `Sector ETFs` (11 ETFs sectoriales).
  - `Article ETFs` (70 ETFs de la lista completa del artículo).
  - Dentro de "Personalizar universo" se pueden quitar ETFs o añadir otros.
- **Backtest de cartera**: varias ETFs, máximo de posiciones configurable (POSQTY),
  tamaño fijo `capital inicial / nº posiciones`, comisión y deslizamiento.
- **Señales al cierre y ejecución a la apertura siguiente** (`SetTradeDelays(1,1,1,1)`,
  `BuyPrice = Open`, `SellPrice = Open`).
- **Indicador RSMK** (`ln(C/índice)` suavizado con EMA × 100) con su media MARS.
- Filtros de entrada del AFL: volumen, RSMK > MARS, RSMK > RSCRIT, máximo de 3 barras,
  RSMK < 25, régimen del índice + correlación, RSI en sobreventa con giro y
  `EMA(5) > MA(MABARS)` desplazada 5 barras.
- Salidas: señal RS/correlación, objetivo de beneficio y salida temporal.
- **Comparación con Buy & Hold del índice** (SPY por defecto) y, opcionalmente,
  cartera equiponderada.
- **Panel central** con velas + indicadores debajo (RSMK/MARS, RSI, correlación,
  volumen y medias).
- **Exportación** de operaciones a CSV y libro XLSX completo (operaciones,
  posiciones abiertas, curva de capital, métricas, rentabilidad anual y parámetros).

## Estructura

```
ALOWRISK/
├── app.py            # Interfaz Streamlit, gráficos y exportación
├── strategy.py       # Indicadores RSMK/RSI/correlación y motor de backtest
├── data.py           # Descarga de Yahoo Finance y normalización
├── requirements.txt
└── .streamlit/config.toml
```

## Notas

- Los datos son diarios y ajustados por dividendos/splits (Yahoo Finance).
- El periodo se descarga con ~550 días extra de calentamiento para que la
  correlación de 100 barras y el RSMK estén disponibles desde la fecha de inicio.
- Los resultados del artículo original dependen del universo de ETFs y del periodo
  exactos; esta app reproduce la lógica, no garantiza cifras idénticas.
- Uso educativo. No es asesoramiento financiero.
