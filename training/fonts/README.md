# Шрифты для синтетических табличек

Открытые шрифты, которыми `training/ocr_lib.py` рисует синтетические номера
(нужны, чтобы модель видела буквы, которых нет на реальных фото).
Взяты из пакетов Ubuntu; лицензии разрешают распространение, тексты в `licenses/`.

| Файл | Шрифт | Лицензия |
|---|---|---|
| DejaVuSans-Bold.ttf | DejaVu Sans | Bitstream Vera / public domain (`licenses/DejaVu.txt`) |
| FreeSansBold.ttf | GNU FreeFont | GPL-3 с исключением для шрифтов (`licenses/FreeFont.txt`) |
| LiberationSans-Bold.ttf, LiberationSans-BoldItalic.ttf | Liberation | SIL OFL 1.1 (`licenses/Liberation.txt`) |
| Ubuntu[wdth,wght].ttf, UbuntuSans[wdth,wght].ttf | Ubuntu | Ubuntu Font Licence 1.0 (`licenses/Ubuntu.txt`) |

Проприетарные шрифты (например, Bahnschrift из Windows, похожий на DIN) сознательно не используются.
