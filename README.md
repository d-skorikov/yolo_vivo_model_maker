# YOLO Model Maker

Обучение модели YOLO (Ultralytics) на разметке, выгруженной из **CVAT**.
Программа берёт zip-архив CVAT, сама раскладывает изображения на обучающую
и проверочную выборки, создаёт `data.yaml` и запускает обучение.
Обученную модель можно сразу экспортировать в формат **OpenVINO**.

## Состав

| Файл               | Назначение                                                           |
|--------------------|----------------------------------------------------------------------|
| `run.bat`          | запуск графического интерфейса (двойной щелчок)                      |
| `yolo_gui.py`      | графический интерфейс                                                |
| `yolo_trainer.py`  | подготовка датасета и обучение (можно запускать из командной строки) |
| `requirements.txt` | зависимости Python                                                   |

## Установка

1. Установите Python 3.9 или новее.
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
   `run.bat` сам предложит это сделать, если `ultralytics` не найден.
3. Для обучения на видеокарте NVIDIA установите PyTorch с поддержкой CUDA
   (команду для своей системы возьмите на https://pytorch.org/get-started/locally/).
   Без этого обучение пойдёт на процессоре — заметно медленнее.

Если рядом с программой есть виртуальное окружение `.venv`, `run.bat` использует его.

## Экспорт из CVAT

В CVAT: **Задача → Actions → Export task dataset**, формат **YOLO 1.1**
(поддерживается и **Ultralytics YOLO Detection**), с изображениями
(*Save images*). Получится zip примерно такого вида:

```
obj.names            ← список классов, по одному в строке
obj.data
train.txt
obj_train_data/
    frame_000000.png
    frame_000000.txt ← разметка YOLO: class x_center y_center width height
    ...
```

## Работа через графический интерфейс

Запустите `run.bat`.

1. **Режим**
   - *Подготовить датасет и обучить* — полный цикл;
   - *Только подготовить датасет* — создать папку датасета без обучения;
   - *Только обучить* — использовать уже подготовленную папку датасета;
   - *Только экспорт в OpenVINO* — преобразовать готовую модель `.pt` без обучения.
2. **Данные**
   - *Архив CVAT* — zip-файл из CVAT;
   - *Папка датасета* — куда разложить данные (по умолчанию `DATASET`);
   - *Доля train* — доля изображений для обучения, остальные идут в проверку (0.8 = 80/20);
   - *Seed* — при одинаковом seed разбиение всегда одинаковое;
   - *Классы* — обычно пусто: имена берутся из `obj.names`.
3. **Параметры обучения** — исходная модель (`yolo11n.pt` … `yolo11x.pt`,
   `yolov8*.pt` или свой `.pt` через «…»), количество эпох, размер изображения,
   batch, устройство (`авто`, `cpu`, `0` — первая видеокарта), workers, patience.
4. **Куда сохранить модель** — папка результатов и имя запуска.
5. **Экспорт в OpenVINO**
   - *Экспортировать модель после обучения* — после обучения `best.pt` будет
     преобразован в OpenVINO;
   - *Точность* — `fp32` (без потерь), `fp16` (вдвое меньше, обычно без потери
     качества), `int8` (самая быстрая на CPU Intel, калибруется по изображениям
     датасета, нужен пакет `nncf`);
   - *Модель .pt* — файл для режима *Только экспорт в OpenVINO*. Для `int8` в этом
     режиме используется `data.yaml` из поля *Папка датасета*.
6. Нажмите **▶ Запустить**. Ход работы отображается в журнале,
   **■ Остановить** прерывает процесс.

Настройки запоминаются в `gui_settings.json` и восстанавливаются при следующем запуске.

## Работа из командной строки

```bash
python yolo_trainer.py --zip dataset.zip
python yolo_trainer.py --zip dataset.zip --model yolo11s.pt --epochs 100 --output-dir models --name white_box
python yolo_trainer.py --zip dataset.zip --prepare-only
python yolo_trainer.py --skip-prepare --dataset-dir DATASET --epochs 30
python yolo_trainer.py --zip dataset.zip --openvino --ov-precision fp16
python yolo_trainer.py --export-only runs/train/weights/best.pt --ov-precision int8 --dataset-dir DATASET
```

Все параметры: `python yolo_trainer.py -h`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--zip` | — | zip-архив CVAT |
| `--dataset-dir` | `DATASET` | папка подготовленного датасета |
| `--train-pct` | `0.8` | доля изображений в train |
| `--seed` | `42` | seed разбиения train/val |
| `--names` | из архива | имена классов вручную |
| `--prepare-only` | — | только подготовить датасет |
| `--skip-prepare` | — | обучать на готовом `--dataset-dir` |
| `--model` | `yolo11m.pt` | исходная модель |
| `--epochs` | `50` | количество эпох |
| `--imgsz` | `640` | размер изображения |
| `--batch` | `16` | размер батча (`-1` — автоподбор) |
| `--device` | авто | `cpu`, `0`, `0,1` |
| `--workers` | `8` | потоки загрузки данных |
| `--patience` | `100` | ранняя остановка без улучшений N эпох |
| `--output-dir` | `runs` | папка результатов |
| `--name` | `train` | имя запуска |
| `--exist-ok` | — | перезаписывать папку запуска |
| `--openvino` | — | после обучения экспортировать `best.pt` в OpenVINO |
| `--ov-precision` | `fp32` | точность OpenVINO: `fp32`, `fp16`, `int8` |
| `--export-only` | — | только экспорт указанного `.pt` в OpenVINO, без обучения |

## Что получается

Подготовленный датасет:

```
DATASET/
    data.yaml
    train/images  train/labels
    val/images    val/labels
```

`data.yaml` создаётся автоматически:

```yaml
path: C:/.../DATASET
train: train/images
val: val/images

nc: 1
names:
  0: 'white_box'
```

Изображения без разметки получают пустой `.txt` и используются как фоновые примеры.

Результаты обучения:

```
runs/
    train/
        weights/best.pt   ← лучшая модель
        weights/last.pt   ← модель последней эпохи
        weights/best_openvino_model/  ← OpenVINO-модель (если включён экспорт)
        results.png, confusion_matrix.png, ...
    train_best.pt         ← копия best.pt для удобства
    train_fp16_openvino_model/  ← копия OpenVINO-модели (в имени — точность)
        best.xml, best.bin, metadata.yaml
```

OpenVINO-модель — это папка с файлами `best.xml` (структура сети) и `best.bin` (веса).
Использовать её можно так же, как `.pt`:

```python
from ultralytics import YOLO
model = YOLO('runs/train_fp16_openvino_model')
results = model('image.jpg')
```

или напрямую через OpenVINO Runtime (`openvino.Core().read_model('best.xml')`).

Если папка запуска уже существует, Ultralytics создаст `train2`, `train3`, …
(или перезапишет её при включённой опции *Перезаписывать папку запуска*).

## Важно

- При каждой подготовке из zip в папке датасета **удаляются и создаются заново**
  `train/`, `val/` и `data.yaml`. Остальные файлы в этой папке не трогаются.
- Исходные модели `yolo11*.pt` / `yolov8*.pt` при первом использовании скачиваются
  Ultralytics автоматически (нужен интернет).
- Если обучение падает с нехваткой памяти видеокарты — уменьшите `batch` или `imgsz`,
  либо возьмите модель поменьше (`n` или `s`).
- Для экспорта в OpenVINO нужен пакет `openvino` (есть в `requirements.txt`),
  для `int8` дополнительно `nncf`: `pip install nncf`. Если пакета нет,
  Ultralytics попытается установить его сам.
- Размер изображения при экспорте берётся из поля *Размер изображения* (`--imgsz`);
  используйте тот же, что при обучении.
