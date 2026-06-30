# Relatório - MAC5915 EP1

Este diretório contém o relatório técnico do projeto de Calibração de Câmera e Realidade Aumentada.

## Estrutura

```
report/
├── relatorio.tex          # Arquivo principal LaTeX
├── sections/             # Seções do relatório
│   ├── introducao.tex
│   ├── metodologia.tex
│   ├── resultados.tex
│   ├── discussao.tex
│   ├── conclusao.tex
│   └── bibliografia.tex
├── figures/              # Figuras e imagens (se necessário)
└── README.md             # Este arquivo
```

## Compilação

### Pré-requisitos

É necessário ter instalado:
- LaTeX (TeX Live, MiKTeX ou MacTeX)
- Pacotes LaTeX: `babel`, `graphicx`, `listings`, `hyperref`, `amsmath`, `booktabs`, `geometry`, `float`, `caption`, `subcaption`

### Compilar o PDF

No diretório `report/`, execute:

```bash
pdflatex relatorio.tex
pdflatex relatorio.tex  # Executar duas vezes para referências cruzadas
```

Ou usando `latexmk` (recomendado):

```bash
latexmk -pdf relatorio.tex
```

### Limpar arquivos auxiliares

```bash
latexmk -c relatorio.tex
```

Ou manualmente:

```bash
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz
```

## Conteúdo

O relatório inclui:

1. **Introdução**: Contexto, objetivos e estrutura
2. **Metodologia**: Detalhes técnicos de cada componente
   - Calibração de câmera
   - AR com cubo
   - AR com bola de tênis
3. **Resultados**: Métricas, tabelas e análise
4. **Discussão**: Análise crítica e limitações
5. **Conclusão**: Resumo e trabalhos futuros

## Figuras

As figuras de calibração são referenciadas do diretório `../out/`. Certifique-se de que os arquivos existem antes de compilar.

## Notas

- O relatório está em português
- As referências às figuras assumem que as imagens estão em `../out/`
- Ajuste os caminhos das figuras se necessário
