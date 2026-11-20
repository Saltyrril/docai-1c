# notebooks/

Ноутбуки для Google Colab: дообучение Qwen3.5-2B (LoRA) и тяжёлые эксперименты,
которые не помещаются в 4 ГБ VRAM ноутбука.

Правило: в ноутбуки не попадают реальные документы из data/real/.

| Файл | Что это |
|---|---|
| `finetune_qwen35_2b.ipynb` | Дообучение Qwen3.5-2B (16-битная LoRA, Unsloth) на датасете `docai finetune build`, экспорт GGUF Q8_0 + mmproj. Инструкция — [docs/FINETUNE.md](../docs/FINETUNE.md), решения — D77 |
| `Modelfile.docai-qwen35-2b-ft` | Modelfile Ollama для результата (такой же ноутбук кладёт рядом с GGUF); подключение — `scripts/finetune_install.ps1` |

Ноутбуки хранятся без вывода ячеек.
