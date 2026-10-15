"""
Обучение модели YOLO на датасете, экспортированном из CVAT.

Что делает программа:
  1. Принимает zip-архив, выгруженный из CVAT (формат "YOLO 1.1":
     obj.names, obj.data, train.txt, obj_train_data/*.png|*.txt,
     либо формат "Ultralytics YOLO": data.yaml, images/..., labels/...).
  2. Перемешивает изображения и делит их на обучающую (train) и
     проверочную (val) выборки.
  3. Создаёт структуру для обучения YOLO:
         <dataset_dir>/
             data.yaml
             train/images  train/labels
             val/images    val/labels
     data.yaml строится по созданным папкам и списку классов из obj.names.
  4. Запускает обучение ultralytics YOLO с заданными параметрами и
     сохраняет результаты в указанную папку.
  5. По желанию экспортирует лучшую модель в формат OpenVINO (FP32/FP16/INT8).

Примеры запуска:
  python yolo_trainer.py --zip dataset.zip
  python yolo_trainer.py --zip dataset.zip --epochs 100 --model yolo11s.pt --output-dir models --name white_box
  python yolo_trainer.py --zip dataset.zip --prepare-only          # только подготовить DATASET
  python yolo_trainer.py --skip-prepare --dataset-dir DATASET      # обучить на уже готовом DATASET
  python yolo_trainer.py --zip dataset.zip --openvino --ov-precision fp16   # обучить и экспортировать в OpenVINO
  python yolo_trainer.py --export-only runs/train/weights/best.pt --ov-precision int8  # только экспорт
"""

import argparse
import random
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SPLITS = ('train', 'val')
OV_PRECISIONS = ('fp32', 'fp16', 'int8')


# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Подготовка датасета из CVAT zip и обучение YOLO',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data = parser.add_argument_group('Данные')
    data.add_argument('--zip', help='zip-архив, выгруженный из CVAT')
    data.add_argument('--dataset-dir', default='DATASET',
                      help='папка, куда будет записан подготовленный датасет')
    data.add_argument('--train-pct', type=float, default=0.8,
                      help='доля изображений для train, остальные идут в val (0.01..0.99)')
    data.add_argument('--seed', type=int, default=42,
                      help='seed для воспроизводимого разбиения train/val')
    data.add_argument('--names', nargs='+',
                      help='имена классов вручную (иначе берутся из obj.names / data.yaml архива)')
    data.add_argument('--prepare-only', action='store_true',
                      help='только подготовить датасет, без обучения')
    data.add_argument('--skip-prepare', action='store_true',
                      help='не распаковывать zip, обучать на уже готовом --dataset-dir')

    train = parser.add_argument_group('Обучение')
    train.add_argument('--model', default='yolo11m.pt',
                       help='исходная модель (yolo11n.pt, yolo11s.pt, yolo11m.pt, ... или путь к своему .pt)')
    train.add_argument('--epochs', type=int, default=50, help='количество эпох')
    train.add_argument('--imgsz', type=int, default=640, help='размер изображения')
    train.add_argument('--batch', type=int, default=16, help='размер батча (-1 = авто)')
    train.add_argument('--device', default=None,
                       help='устройство: 0, 0,1, cpu (по умолчанию выбирается автоматически)')
    train.add_argument('--workers', type=int, default=8, help='число потоков загрузки данных')
    train.add_argument('--patience', type=int, default=100,
                       help='остановка, если метрика не улучшается N эпох')

    out = parser.add_argument_group('Результат')
    out.add_argument('--output-dir', default='runs',
                     help='папка, куда записываются результаты обучения и модель')
    out.add_argument('--name', default='train',
                     help='имя запуска (подпапка в --output-dir)')
    out.add_argument('--exist-ok', action='store_true',
                     help='перезаписывать существующую папку запуска вместо создания train2, train3, ...')

    ov = parser.add_argument_group('Экспорт в OpenVINO')
    ov.add_argument('--openvino', action='store_true',
                    help='после обучения экспортировать best.pt в формат OpenVINO')
    ov.add_argument('--ov-precision', choices=OV_PRECISIONS, default='fp32',
                    help='точность OpenVINO-модели (int8 требует пакет nncf и датасет для калибровки)')
    ov.add_argument('--export-only', metavar='MODEL_PT',
                    help='только экспортировать указанный .pt в OpenVINO, без подготовки и обучения')

    args = parser.parse_args()

    if args.export_only:
        if not Path(args.export_only).is_file():
            parser.error(f'файл модели не найден: {args.export_only}')
        return args
    if not args.skip_prepare and not args.zip:
        parser.error('укажите --zip (или --skip-prepare для обучения на готовом датасете)')
    if args.prepare_only and args.skip_prepare:
        parser.error('--prepare-only и --skip-prepare взаимоисключающие')
    if not 0.01 <= args.train_pct <= 0.99:
        parser.error('--train-pct должен быть в диапазоне 0.01..0.99')
    return args


# ---------------------------------------------------------------------------
# Чтение архива CVAT
# ---------------------------------------------------------------------------

def read_class_names(zf, names):
    """Список классов: obj.names (YOLO 1.1) или data.yaml (Ultralytics YOLO)."""
    by_name = {PurePosixPath(n).name: n for n in names}

    if 'obj.names' in by_name:
        text = zf.read(by_name['obj.names']).decode('utf-8-sig')
        classes = [line.strip() for line in text.splitlines() if line.strip()]
        if classes:
            return classes

    for yaml_name in ('data.yaml', 'dataset.yaml'):
        if yaml_name in by_name:
            import yaml
            cfg = yaml.safe_load(zf.read(by_name[yaml_name]).decode('utf-8-sig')) or {}
            classes = cfg.get('names')
            if isinstance(classes, dict):
                return [classes[k] for k in sorted(classes, key=int)]
            if isinstance(classes, list) and classes:
                return classes

    return None


def find_label(image_name, txt_entries):
    """Ищет txt-разметку для изображения: рядом с ним или в соседней папке labels."""
    img = PurePosixPath(image_name)
    candidates = [str(img.with_suffix('.txt'))]
    parts = list(img.parts)
    if 'images' in parts:
        idx = len(parts) - 1 - parts[::-1].index('images')
        parts[idx] = 'labels'
        candidates.append(str(PurePosixPath(*parts).with_suffix('.txt')))
    for cand in candidates:
        if cand in txt_entries:
            return cand
    return None


def collect_pairs(zf):
    """Возвращает список (имя изображения в архиве, имя txt или None)."""
    names = [n for n in zf.namelist() if not n.endswith('/')]
    txt_entries = {n for n in names if n.lower().endswith('.txt')}
    images = sorted(n for n in names if PurePosixPath(n).suffix.lower() in IMAGE_EXTENSIONS)
    return [(img, find_label(img, txt_entries)) for img in images]


# ---------------------------------------------------------------------------
# Подготовка датасета
# ---------------------------------------------------------------------------

def clean_dataset_dir(dataset_dir):
    """Удаляет только то, что создаёт эта программа: train/, val/, data.yaml."""
    for split in SPLITS:
        split_dir = dataset_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
    yaml_path = dataset_dir / 'data.yaml'
    if yaml_path.exists():
        yaml_path.unlink()


def unique_filename(name, used):
    """Имя файла без пути; если такое уже есть (из разных папок архива) — добавляет суффикс."""
    path = PurePosixPath(name)
    candidate, i = path.name, 1
    while candidate.lower() in used:
        candidate = f'{path.stem}_{i}{path.suffix}'
        i += 1
    used.add(candidate.lower())
    return candidate


def copy_pair(zf, image_name, label_name, out_name, split_dir, num_classes, stats):
    img_dst = split_dir / 'images' / out_name
    lbl_dst = split_dir / 'labels' / (Path(out_name).stem + '.txt')

    with zf.open(image_name) as src, open(img_dst, 'wb') as dst:
        shutil.copyfileobj(src, dst)

    if label_name is None:
        # Изображение без разметки — фоновый пример (пустой файл меток)
        lbl_dst.write_text('', encoding='utf-8')
        stats['no_label'] += 1
        return

    text = zf.read(label_name).decode('utf-8-sig')
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        stats['objects'] += 1
        try:
            cls = int(parts[0])
        except ValueError:
            stats['bad_lines'] += 1
            continue
        if num_classes is not None and not 0 <= cls < num_classes:
            stats['bad_class'] += 1
    if not text.strip():
        stats['empty_label'] += 1
    lbl_dst.write_text(text, encoding='utf-8')


def write_data_yaml(dataset_dir, class_names):
    """Создаёт data.yaml по структуре папок датасета и списку классов."""
    lines = [
        f'path: {dataset_dir.resolve().as_posix()}  # Абсолютный путь',
    ]
    for split in SPLITS:
        if (dataset_dir / split / 'images').is_dir():
            lines.append(f'{split}: {split}/images')
    lines += ['', f'nc: {len(class_names)}', 'names:']
    for i, name in enumerate(class_names):
        safe = str(name).replace("'", "''")
        lines.append(f"  {i}: '{safe}'")
    yaml_path = dataset_dir / 'data.yaml'
    yaml_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return yaml_path


def prepare_dataset(args):
    zip_path = Path(args.zip)
    if not zip_path.is_file():
        sys.exit(f'Файл не найден: {zip_path}')
    dataset_dir = Path(args.dataset_dir)

    with zipfile.ZipFile(zip_path) as zf:
        class_names = args.names or read_class_names(zf, zf.namelist())
        if not class_names:
            sys.exit('Не удалось найти список классов (obj.names / data.yaml) в архиве. '
                     'Укажите их вручную через --names.')

        pairs = collect_pairs(zf)
        if not pairs:
            sys.exit('В архиве не найдено изображений.')

        random.Random(args.seed).shuffle(pairs)
        train_num = int(len(pairs) * args.train_pct)
        train_num = min(max(train_num, 1), len(pairs) - 1) if len(pairs) > 1 else len(pairs)
        split_pairs = {'train': pairs[:train_num], 'val': pairs[train_num:]}

        print(f'Архив:          {zip_path}')
        print(f'Классы ({len(class_names)}):    {", ".join(class_names)}')
        print(f'Изображений:    {len(pairs)}  (train: {len(split_pairs["train"])}, '
              f'val: {len(split_pairs["val"])})')

        clean_dataset_dir(dataset_dir)
        stats = dict(objects=0, no_label=0, empty_label=0, bad_lines=0, bad_class=0)
        used_names = set()
        for split, items in split_pairs.items():
            split_dir = dataset_dir / split
            (split_dir / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
            for image_name, label_name in items:
                out_name = unique_filename(image_name, used_names)
                copy_pair(zf, image_name, label_name, out_name, split_dir, len(class_names), stats)

    yaml_path = write_data_yaml(dataset_dir, class_names)

    print(f'Объектов:       {stats["objects"]}')
    if stats['no_label'] or stats['empty_label']:
        print(f'Без разметки:   {stats["no_label"] + stats["empty_label"]} (используются как фон)')
    if stats['bad_lines'] or stats['bad_class']:
        print(f'ВНИМАНИЕ: некорректных строк: {stats["bad_lines"]}, '
              f'классов вне диапазона 0..{len(class_names) - 1}: {stats["bad_class"]}')
    print(f'Датасет готов:  {dataset_dir.resolve()}')
    print(f'data.yaml:      {yaml_path.resolve()}')
    return yaml_path


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train(args, data_yaml):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit('Пакет ultralytics не установлен: pip install ultralytics')

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(Path(data_yaml).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        project=str(output_dir),
        name=args.name,
        exist_ok=args.exist_ok,
    )
    if args.device is not None:
        train_kwargs['device'] = args.device

    model.train(**train_kwargs)

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / 'weights' / 'best.pt'
    print(f'\nРезультаты обучения: {run_dir}')
    if not best.is_file():
        return
    final = output_dir / f'{run_dir.name}_best.pt'
    shutil.copy2(best, final)
    print(f'Лучшая модель:       {best}')
    print(f'Копия модели:        {final}')

    if args.openvino:
        export_openvino(args, best, output_dir, run_dir.name, data_yaml)


def export_openvino(args, weights, output_dir, base_name, data_yaml=None):
    """Экспорт .pt в OpenVINO; копия модели кладётся в output_dir/<base_name>_<precision>_openvino_model."""
    precision = args.ov_precision
    export_kwargs = dict(format='openvino', imgsz=args.imgsz,
                         half=precision == 'fp16', int8=precision == 'int8')
    if precision == 'int8':
        # Для INT8 нужна калибровка на изображениях датасета
        if data_yaml is None or not Path(data_yaml).is_file():
            sys.exit('Для INT8 нужен датасет для калибровки: укажите --dataset-dir с data.yaml '
                     'или выберите --ov-precision fp16/fp32.')
        export_kwargs['data'] = str(Path(data_yaml).resolve())

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit('Пакет ultralytics не установлен: pip install ultralytics')

    print(f'\nЭкспорт в OpenVINO ({precision.upper()}): {weights}')
    exported = Path(YOLO(str(weights)).export(**export_kwargs))

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Окончание _openvino_model нужно, чтобы YOLO('<папка>') распознал формат
    final = output_dir / f'{base_name}_{precision}_openvino_model'
    if final.resolve() != exported.resolve():
        if final.exists():
            shutil.rmtree(final)
        shutil.copytree(exported, final)
    print(f'OpenVINO-модель:     {exported}')
    print(f'Копия модели:        {final}')


def main():
    args = parse_args()

    if args.export_only:
        weights = Path(args.export_only)
        data_yaml = Path(args.dataset_dir) / 'data.yaml'
        export_openvino(args, weights, args.output_dir, weights.stem,
                        data_yaml if data_yaml.is_file() else None)
        return

    if args.skip_prepare:
        data_yaml = Path(args.dataset_dir) / 'data.yaml'
        if not data_yaml.is_file():
            sys.exit(f'Не найден {data_yaml}. Сначала подготовьте датасет через --zip.')
    else:
        data_yaml = prepare_dataset(args)

    if not args.prepare_only:
        train(args, data_yaml)


if __name__ == '__main__':
    main()
