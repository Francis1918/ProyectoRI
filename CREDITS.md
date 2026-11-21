# Créditos y licencias de terceros

## Fotografía de fondo (portada de Lente)

- **Obra:** «Crazy Mountains, Montana» (`File:Crazy Mountains, Montana (31907528007).jpg`).
- **Autoría:** Janusz Sobolewski (Mississauga, Ontario, Canadá).
- **Origen:** Wikimedia Commons, <https://commons.wikimedia.org/wiki/File:Crazy_Mountains,_Montana_(31907528007).jpg>
- **Licencia:** Creative Commons Atribución 2.0 (CC BY 2.0), <https://creativecommons.org/licenses/by/2.0/>
- **Cambios realizados** (`scripts/prepare_background.py`): redimensionada a 2560, 1600 y 900 px
  de ancho (versión móvil recortada en vertical), mayor luminosidad en sombras y medios tonos,
  desplazamiento de tono ~7° hacia el violeta, saturación de los azules ligeramente menor y un
  resplandor suave rosado en la base de las nubes (mezcla «screen»). No es la imagen original.
- La imagen `img.png` de la raíz del repositorio es solo la maqueta de referencia visual; **no** se
  usa en el sitio.
- Se muestra una atribución visible en el pie de la portada.

## Fuentes tipográficas (alojadas en `public/fonts/`)

- **Fraunces** (peso 300, subconjunto latino), © The Fraunces Project Authors / Undercase Type. SIL OFL 1.1.
- **Hanken Grotesk** (400–700, subconjunto latino), © The Hanken Grotesk Project Authors. SIL OFL 1.1.
- Obtenidas de Google Fonts.

## Modelo y datos

- **all-MiniLM-L6-v2** (`sentence-transformers/all-MiniLM-L6-v2`, revisión
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), Apache 2.0. Se usa su grafo ONNX oficial
  (`onnx/model.onnx`) y su `vocab.txt`.
- **FiQA** (`BeIR/fiqa`, conjunto FiQA-2018, preguntas y respuestas financieras), distribuido por
  BEIR. Consulta la licencia del conjunto en su ficha de Hugging Face.
