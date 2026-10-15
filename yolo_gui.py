"""
Графический интерфейс для yolo_trainer.py.

Собирает параметры в форме, запускает yolo_trainer.py отдельным процессом
и показывает его вывод в окне журнала. Последние настройки сохраняются
в gui_settings.json рядом со скриптом.
"""

import codecs
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

APP_DIR = Path(__file__).resolve().parent
TRAINER = APP_DIR / 'yolo_trainer.py'
SETTINGS_FILE = APP_DIR / 'gui_settings.json'

MODELS = [
    'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
]
DEVICES = ['авто', 'cpu', '0', '0,1']
MODES = {
    'full': 'Подготовить датасет и обучить',
    'prepare': 'Только подготовить датасет',
    'train': 'Только обучить (датасет уже готов)',
    'export': 'Только экспорт в OpenVINO',
}
OV_PRECISIONS = ['fp32', 'fp16', 'int8']

DEFAULTS = {
    'zip': '',
    'dataset_dir': str(APP_DIR / 'DATASET'),
    'train_pct': 0.8,
    'seed': 42,
    'names': '',
    'mode': 'full',
    'model': 'yolo11m.pt',
    'epochs': 50,
    'imgsz': 640,
    'batch': 16,
    'device': 'авто',
    'workers': 8,
    'patience': 100,
    'output_dir': str(APP_DIR / 'runs'),
    'name': 'train',
    'exist_ok': False,
    'openvino': False,
    'ov_precision': 'fp32',
    'export_model': '',
}

ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')


def python_executable():
    """Интерпретатор для дочернего процесса: python.exe вместо pythonw.exe, чтобы был вывод."""
    exe = Path(sys.executable)
    if exe.name.lower() == 'pythonw.exe':
        console = exe.with_name('python.exe')
        if console.is_file():
            return str(console)
    return str(exe)


class TrainerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('YOLO Model Maker — обучение по датасету CVAT')
        self.minsize(820, 640)

        self.process = None
        self.output_queue = queue.Queue()
        self.vars = {}

        self._build_ui()
        self._load_settings()
        self._on_mode_change()
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self.after(100, self._poll_output)

    # ------------------------------------------------------------------ UI

    def _var(self, key, kind=tk.StringVar):
        var = kind(value=DEFAULTS[key])
        self.vars[key] = var
        return var

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill='both', expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(3, weight=1)

        # --- Режим
        mode_box = ttk.LabelFrame(root, text='Режим', padding=8)
        mode_box.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        mode_var = self._var('mode')
        for i, (key, text) in enumerate(MODES.items()):
            ttk.Radiobutton(mode_box, text=text, value=key, variable=mode_var,
                            command=self._on_mode_change).grid(row=0, column=i, padx=(0, 16))

        # --- Данные
        data = ttk.LabelFrame(root, text='Данные', padding=8)
        data.grid(row=1, column=0, sticky='nsew', padx=(0, 4))
        data.columnconfigure(1, weight=1)

        self.zip_widgets = self._path_row(data, 0, 'Архив CVAT (zip):', 'zip', self._browse_zip)
        self._path_row(data, 1, 'Папка датасета:', 'dataset_dir',
                       lambda: self._browse_dir('dataset_dir'))
        self.split_widgets = [
            *self._spin_row(data, 2, 'Доля train:', 'train_pct', 0.05, 0.95, 0.05, tk.DoubleVar),
            *self._spin_row(data, 3, 'Seed разбиения:', 'seed', 0, 999999, 1, tk.IntVar),
        ]
        ttk.Label(data, text='Классы (через пробел):').grid(row=4, column=0, sticky='w', pady=2)
        names_entry = ttk.Entry(data, textvariable=self._var('names'))
        names_entry.grid(row=4, column=1, columnspan=2, sticky='ew', pady=2)
        ttk.Label(data, text='пусто — взять из obj.names архива', foreground='gray') \
            .grid(row=5, column=1, columnspan=2, sticky='w')
        self.split_widgets.append(names_entry)

        # --- Обучение
        train = ttk.LabelFrame(root, text='Параметры обучения', padding=8)
        train.grid(row=1, column=1, sticky='nsew', padx=(4, 0))
        train.columnconfigure(1, weight=1)

        ttk.Label(train, text='Исходная модель:').grid(row=0, column=0, sticky='w', pady=2)
        model_box = ttk.Combobox(train, textvariable=self._var('model'), values=MODELS)
        model_box.grid(row=0, column=1, sticky='ew', pady=2)
        ttk.Button(train, text='…', width=3, command=self._browse_model) \
            .grid(row=0, column=2, padx=(4, 0))
        self.train_widgets = [model_box]
        self.train_widgets += self._spin_row(train, 1, 'Эпох:', 'epochs', 1, 10000, 1, tk.IntVar)
        self.imgsz_widgets = self._spin_row(train, 2, 'Размер изображения:', 'imgsz', 32, 4096, 32, tk.IntVar)
        self.train_widgets += self._spin_row(train, 3, 'Batch (-1 = авто):', 'batch', -1, 1024, 1, tk.IntVar)
        ttk.Label(train, text='Устройство:').grid(row=4, column=0, sticky='w', pady=2)
        device_box = ttk.Combobox(train, textvariable=self._var('device'), values=DEVICES)
        device_box.grid(row=4, column=1, sticky='ew', pady=2)
        self.train_widgets.append(device_box)
        self.train_widgets += self._spin_row(train, 5, 'Workers:', 'workers', 0, 64, 1, tk.IntVar)
        self.train_widgets += self._spin_row(train, 6, 'Patience:', 'patience', 0, 10000, 1, tk.IntVar)

        # --- Результат
        out = ttk.LabelFrame(root, text='Куда сохранить модель', padding=8)
        out.grid(row=2, column=0, sticky='nsew', pady=6, padx=(0, 4))
        out.columnconfigure(1, weight=1)
        self.output_widgets = self._path_row(out, 0, 'Папка результатов:', 'output_dir',
                                             lambda: self._browse_dir('output_dir'))
        ttk.Label(out, text='Имя запуска:').grid(row=1, column=0, sticky='w', pady=2)
        name_entry = ttk.Entry(out, textvariable=self._var('name'))
        name_entry.grid(row=1, column=1, sticky='ew', pady=2)
        exist_ok = ttk.Checkbutton(out, text='Перезаписывать папку запуска',
                                   variable=self._var('exist_ok', tk.BooleanVar))
        exist_ok.grid(row=2, column=1, sticky='w')
        self.train_widgets += [name_entry, exist_ok]

        # --- OpenVINO
        ov = ttk.LabelFrame(root, text='Экспорт в OpenVINO', padding=8)
        ov.grid(row=2, column=1, sticky='nsew', pady=6, padx=(4, 0))
        ov.columnconfigure(1, weight=1)
        self.ov_check = ttk.Checkbutton(ov, text='Экспортировать модель после обучения',
                                        variable=self._var('openvino', tk.BooleanVar),
                                        command=self._on_mode_change)
        self.ov_check.grid(row=0, column=0, columnspan=3, sticky='w')
        ttk.Label(ov, text='Точность:').grid(row=1, column=0, sticky='w', pady=2)
        self.ov_precision = ttk.Combobox(ov, textvariable=self._var('ov_precision'),
                                         values=OV_PRECISIONS, state='readonly')
        self.ov_precision.grid(row=1, column=1, columnspan=2, sticky='ew', pady=2)
        self.export_widgets = self._path_row(ov, 2, 'Модель .pt:', 'export_model', self._browse_export_model)
        ttk.Label(ov, text='fp16 — меньше и быстрее; int8 — калибровка по датасету (нужен nncf)',
                  foreground='gray').grid(row=3, column=0, columnspan=3, sticky='w')

        # --- Журнал
        log_box = ttk.LabelFrame(root, text='Журнал', padding=4)
        log_box.grid(row=3, column=0, columnspan=2, sticky='nsew')
        self.log = ScrolledText(log_box, height=14, wrap='none', font=('Consolas', 9),
                                state='disabled')
        self.log.pack(fill='both', expand=True)

        # --- Кнопки
        buttons = ttk.Frame(root)
        buttons.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(6, 0))
        self.start_btn = ttk.Button(buttons, text='▶ Запустить', command=self._start)
        self.start_btn.pack(side='left')
        self.stop_btn = ttk.Button(buttons, text='■ Остановить', command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=6)
        ttk.Button(buttons, text='Открыть папку результатов', command=self._open_output) \
            .pack(side='left')
        ttk.Button(buttons, text='Очистить журнал', command=self._clear_log).pack(side='left', padx=6)
        self.status = ttk.Label(buttons, text='Готово к запуску')
        self.status.pack(side='right')

    def _path_row(self, parent, row, label, key, browse):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=2)
        entry = ttk.Entry(parent, textvariable=self._var(key))
        entry.grid(row=row, column=1, sticky='ew', pady=2)
        button = ttk.Button(parent, text='…', width=3, command=browse)
        button.grid(row=row, column=2, padx=(4, 0))
        return [entry, button]

    def _spin_row(self, parent, row, label, key, lo, hi, step, kind):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=2)
        spin = ttk.Spinbox(parent, from_=lo, to=hi, increment=step, textvariable=self._var(key, kind))
        spin.grid(row=row, column=1, columnspan=2, sticky='ew', pady=2)
        return [spin]

    def _on_mode_change(self):
        mode = self.vars['mode'].get()
        trains = mode in ('full', 'train')
        enabled = [
            (self.zip_widgets + self.split_widgets, mode in ('full', 'prepare')),
            (self.train_widgets + [self.ov_check], trains),
            (self.imgsz_widgets + self.output_widgets, mode != 'prepare'),
            (self.export_widgets, mode == 'export'),
        ]
        for widgets, on in enabled:
            for w in widgets:
                w.configure(state='normal' if on else 'disabled')
        ov_on = mode == 'export' or (trains and self.vars['openvino'].get())
        self.ov_precision.configure(state='readonly' if ov_on else 'disabled')

    # ------------------------------------------------------------- Диалоги

    def _browse_zip(self):
        path = filedialog.askopenfilename(title='Архив CVAT', filetypes=[('ZIP', '*.zip'), ('Все', '*.*')],
                                          initialdir=self._initial_dir('zip'))
        if path:
            self.vars['zip'].set(path)

    def _browse_model(self):
        path = filedialog.askopenfilename(title='Исходная модель', filetypes=[('PyTorch', '*.pt'), ('Все', '*.*')],
                                          initialdir=str(APP_DIR))
        if path:
            self.vars['model'].set(path)

    def _browse_export_model(self):
        path = filedialog.askopenfilename(title='Модель для экспорта', filetypes=[('PyTorch', '*.pt'), ('Все', '*.*')],
                                          initialdir=self._initial_dir('output_dir'))
        if path:
            self.vars['export_model'].set(path)

    def _browse_dir(self, key):
        path = filedialog.askdirectory(title='Выберите папку', initialdir=self._initial_dir(key))
        if path:
            self.vars[key].set(path)

    def _initial_dir(self, key):
        current = Path(self.vars[key].get() or APP_DIR)
        while not current.is_dir() and current != current.parent:
            current = current.parent
        return str(current if current.is_dir() else APP_DIR)

    # ------------------------------------------------------------ Настройки

    def _collect(self):
        values = {}
        for key, var in self.vars.items():
            try:
                values[key] = var.get()
            except tk.TclError:
                raise ValueError(f'Некорректное значение в поле «{key}»')
        return values

    def _load_settings(self):
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        for key, value in saved.items():
            if key in self.vars:
                try:
                    self.vars[key].set(value)
                except tk.TclError:
                    pass

    def _save_settings(self):
        try:
            SETTINGS_FILE.write_text(json.dumps(self._collect(), ensure_ascii=False, indent=2),
                                     encoding='utf-8')
        except (OSError, ValueError):
            pass

    # -------------------------------------------------------------- Запуск

    def _build_command(self, v):
        mode = v['mode']
        cmd = [python_executable(), '-u', str(TRAINER), '--dataset-dir', v['dataset_dir']]

        if mode == 'export':
            model = v['export_model'].strip()
            if not model or not Path(model).is_file():
                raise ValueError('Укажите существующий файл модели .pt для экспорта.')
            return cmd + ['--export-only', model, '--ov-precision', v['ov_precision'],
                          '--imgsz', str(v['imgsz']), '--output-dir', v['output_dir']]

        if mode == 'train':
            cmd.append('--skip-prepare')
        else:
            if not v['zip'] or not Path(v['zip']).is_file():
                raise ValueError('Укажите существующий zip-архив CVAT.')
            if not 0.01 <= v['train_pct'] <= 0.99:
                raise ValueError('Доля train должна быть в диапазоне 0.01..0.99.')
            cmd += ['--zip', v['zip'], '--train-pct', str(v['train_pct']), '--seed', str(v['seed'])]
            if v['names'].strip():
                cmd += ['--names', *v['names'].split()]

        if mode == 'prepare':
            cmd.append('--prepare-only')
        else:
            if not v['model'].strip():
                raise ValueError('Укажите исходную модель.')
            if v['epochs'] < 1:
                raise ValueError('Количество эпох должно быть не меньше 1.')
            cmd += ['--model', v['model'].strip(), '--epochs', str(v['epochs']),
                    '--imgsz', str(v['imgsz']), '--batch', str(v['batch']),
                    '--workers', str(v['workers']), '--patience', str(v['patience']),
                    '--output-dir', v['output_dir'], '--name', v['name'].strip() or 'train']
            device = v['device'].strip()
            if device and device != 'авто':
                cmd += ['--device', device]
            if v['exist_ok']:
                cmd.append('--exist-ok')
            if v['openvino']:
                cmd += ['--openvino', '--ov-precision', v['ov_precision']]
        return cmd

    def _start(self):
        if self.process:
            return
        try:
            values = self._collect()
            cmd = self._build_command(values)
        except ValueError as e:
            messagebox.showerror('Ошибка параметров', str(e), parent=self)
            return
        self._save_settings()

        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self._append('$ ' + subprocess.list2cmdline(cmd) + '\n\n')
        try:
            self.process = subprocess.Popen(
                cmd, cwd=str(APP_DIR), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)
        except OSError as e:
            messagebox.showerror('Ошибка запуска', str(e), parent=self)
            return

        threading.Thread(target=self._reader, args=(self.process,), daemon=True).start()
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.status.configure(text='Выполняется…')

    def _reader(self, proc):
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            self.output_queue.put(('text', decoder.decode(chunk)))
        self.output_queue.put(('text', decoder.decode(b'', final=True)))
        self.output_queue.put(('exit', proc.wait()))

    def _stop(self):
        if not self.process:
            return
        if os.name == 'nt':
            # /T — вместе с дочерними процессами (загрузчики данных)
            subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            self.process.terminate()
        self.status.configure(text='Остановка…')

    def _poll_output(self):
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == 'text':
                    self._append(payload)
                else:
                    self._finished(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_output)

    def _finished(self, code):
        self.process = None
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        if code == 0:
            self.status.configure(text='Завершено успешно')
            self._append('\n=== Завершено успешно ===\n')
        else:
            self.status.configure(text=f'Завершено с ошибкой (код {code})')
            self._append(f'\n=== Процесс завершён с кодом {code} ===\n')

    # -------------------------------------------------------------- Журнал

    def _append(self, text):
        """Добавляет текст в журнал; '\\r' перезаписывает текущую строку (прогресс-бары)."""
        text = ANSI_RE.sub('', text).replace('\r\n', '\n')
        at_bottom = self.log.yview()[1] >= 0.999
        self.log.configure(state='normal')
        for part in re.split(r'(\r|\n)', text):
            if part == '\n':
                self.log.insert('end', '\n')
            elif part == '\r':
                self.log.delete('end-1c linestart', 'end-1c')
            elif part:
                self.log.insert('end', part)
        # Ограничиваем размер журнала
        lines = int(self.log.index('end-1c').split('.')[0])
        if lines > 5000:
            self.log.delete('1.0', f'{lines - 5000}.0')
        self.log.configure(state='disabled')
        if at_bottom:
            self.log.see('end')

    def _clear_log(self):
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.configure(state='disabled')

    def _open_output(self):
        path = Path(self.vars['output_dir'].get() or APP_DIR / 'runs')
        if not path.is_dir():
            messagebox.showinfo('Папка результатов', f'Папка ещё не создана:\n{path}', parent=self)
            return
        if os.name == 'nt':
            os.startfile(path)
        else:
            subprocess.Popen(['xdg-open', str(path)])

    def _on_close(self):
        if self.process:
            if not messagebox.askyesno('Выход', 'Процесс ещё выполняется. Остановить и выйти?', parent=self):
                return
            self._stop()
        self._save_settings()
        self.destroy()


if __name__ == '__main__':
    TrainerGUI().mainloop()
