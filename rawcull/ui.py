"""RawCull desktop UI (PySide6 / Qt 6).

Layout:  [ thumbnail grid ] [ zoomable preview with focus overlay ] [ scores + actions ]

Keyboard:  arrows move · P pick · X reject · U unflag · 0-5 rating · [ / ] previous / next group
           F fit · Z toggle 100% (loads the full RAW) · O toggle overlays · G only this group
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, QSettings, QSize, Qt, QThread, Signal
from PySide6.QtGui import (QAction, QColor, QFont, QIcon, QImage, QKeySequence, QPainter,
                           QPalette, QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QFrame, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
                               QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QSplitter,
                               QToolBar, QVBoxLayout, QWidget)

from . import __version__
from . import exif as exif_mod
from .analysis import faces_available
from .config import Settings
from .loader import load_fullres_rgb, load_preview
from .pipeline import (export_csv, export_picks, run_analysis, save_decisions, write_xmp_all)
from .scoring import effective

GREEN, RED, AMBER, GREY = QColor("#3fb950"), QColor("#f85149"), QColor("#d29922"), QColor("#8b949e")
ICON_W, ICON_H = 168, 132


def bgr_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()


def rgb_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


# ------------------------------------------------------------------------ workers ------
class AnalysisThread(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, folder: str, settings: Settings, force: bool, parent=None):
        super().__init__(parent)
        self.folder, self.settings, self.force = folder, settings, force
        self.cancel_event = threading.Event()

    def run(self):
        try:
            res = run_analysis(self.folder, self.settings,
                               lambda d, t, n: self.progress.emit(d, t, n),
                               self.cancel_event, self.force)
            if not self.cancel_event.is_set():
                self.finished_ok.emit(res)
        except Exception as exc:  # surfaced to the user in a dialog
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class FullResThread(QThread):
    done = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            self.done.emit(self.path, load_fullres_rgb(self.path))
        except Exception as exc:
            self.failed.emit(self.path, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------------ preview ------
class PreviewView(QGraphicsView):
    """Zoomable image view with the autofocus box and detected faces drawn on top."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QColor("#161b22"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._pix = None
        self._overlay_items: list = []
        self._roi = None
        self._faces: list = []
        self._source = ""
        self.overlays_on = True
        self.fit_mode = True
        self._message = None

    def clear_image(self, text: str = ""):
        self._scene.clear()
        self._pix = None
        self._overlay_items = []
        if text:
            self._message = self._scene.addText(text)
            self._message.setDefaultTextColor(QColor("#8b949e"))

    def set_image(self, qimg: QImage):
        self._scene.clear()
        self._overlay_items = []
        self._message = None
        self._pix = self._scene.addPixmap(QPixmap.fromImage(qimg))
        self._scene.setSceneRect(QRectF(self._pix.pixmap().rect()))
        self._rebuild_overlays()
        if self.fit_mode:
            self.fit()

    def set_overlays(self, roi, faces, source):
        self._roi, self._faces, self._source = roi, faces, source
        self._rebuild_overlays()

    def _rebuild_overlays(self):
        for it in self._overlay_items:
            self._scene.removeItem(it)
        self._overlay_items = []
        if self._pix is None or not self.overlays_on:
            return
        r = self._pix.pixmap().rect()
        W, H = r.width(), r.height()

        def add(norm, color, width):
            x, y, w, h = norm
            item = QGraphicsRectItem(x * W, y * H, w * W, h * H)
            pen = QPen(color, width)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setZValue(10)
            self._scene.addItem(item)
            self._overlay_items.append(item)

        for f in self._faces:
            add(f, AMBER, 1.5)
        if self._roi is not None:
            add(self._roi, GREEN, 2.5)

    def toggle_overlays(self):
        self.overlays_on = not self.overlays_on
        self._rebuild_overlays()

    def fit(self):
        self.fit_mode = True
        if self._pix is not None:
            self.resetTransform()
            self.fitInView(self._pix, Qt.AspectRatioMode.KeepAspectRatio)

    def show_100(self):
        """Native pixel size, centred on the focus box (or the image centre)."""
        if self._pix is None:
            return
        self.fit_mode = False
        self.resetTransform()
        r = self._pix.pixmap().rect()
        if self._roi:
            cx, cy = self._roi[0] + self._roi[2] / 2, self._roi[1] + self._roi[3] / 2
        else:
            cx = cy = 0.5
        self.centerOn(cx * r.width(), cy * r.height())

    def wheelEvent(self, ev):
        if self._pix is None:
            return
        self.fit_mode = False
        factor = 1.18 if ev.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.fit_mode:
            self.fit()

    def mouseDoubleClickEvent(self, ev):
        self.fit() if not self.fit_mode else self.show_100()


# -------------------------------------------------------------------------- window ------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"RawCull {__version__} - find the best shots")
        self.resize(1480, 880)
        self.settings_store = QSettings("rawcull", "rawcull")
        self.settings = Settings()
        self.folder: str | None = None
        self.results: list[dict] = []
        self.view_rows: list[dict] = []          # results currently shown in the grid
        self.thumb_cache: dict[str, QPixmap] = {}
        self.preview_cache: dict[str, object] = {}
        self.current: dict | None = None
        self.group_filter: int | None = None
        self.worker: AnalysisThread | None = None
        self.full_thread: FullResThread | None = None
        self.zoom100 = False

        self._build_ui()
        self._build_actions()
        self._update_state()

    # ---- construction ----------------------------------------------------------------
    def _build_ui(self):
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setIconSize(QSize(ICON_W, ICON_H))
        self.grid.setGridSize(QSize(ICON_W + 12, ICON_H + 10))
        self.grid.setSpacing(2)
        self.grid.setMinimumWidth(380)
        self.grid.currentRowChanged.connect(self._on_row_changed)

        self.preview = PreviewView()
        self.preview.clear_image("Open a folder of photos to begin  (Ctrl+O)")
        self.legend = QLabel("<span style='color:#3fb950'>■</span> focus area measured &nbsp; "
                             "<span style='color:#d29922'>■</span> detected faces &nbsp;&nbsp; "
                             "double-click / Z = 100% · wheel = zoom · F = fit · O = overlays")
        self.legend.setStyleSheet("color:#8b949e; padding:4px 8px;")
        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.preview, 1)
        cl.addWidget(self.legend)

        side = QWidget()
        side.setMinimumWidth(300)
        side.setMaximumWidth(380)
        sl = QVBoxLayout(side)
        self.name_lbl = QLabel("-")
        self.name_lbl.setFont(QFont("", 12, QFont.Weight.Bold))
        self.name_lbl.setWordWrap(True)
        self.score_lbl = QLabel("")
        self.score_lbl.setFont(QFont("", 30, QFont.Weight.Bold))
        self.state_lbl = QLabel("")
        self.exif_lbl = QLabel("")
        self.exif_lbl.setStyleSheet("color:#8b949e")
        sl.addWidget(self.name_lbl)
        sl.addWidget(self.score_lbl)
        sl.addWidget(self.state_lbl)
        sl.addWidget(self.exif_lbl)

        form = QFormLayout()
        self.bars = {}
        for key, label in (("focus", "Focus"), ("exposure", "Exposure"), ("face", "Face / eyes")):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            self.bars[key] = bar
            form.addRow(label, bar)
        sl.addLayout(form)
        self.group_lbl = QLabel("")
        self.group_lbl.setWordWrap(True)
        self.reason_lbl = QLabel("")
        self.reason_lbl.setWordWrap(True)
        self.reason_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        sl.addWidget(self.group_lbl)
        sl.addWidget(self.reason_lbl, 1)

        row = QHBoxLayout()
        self.btn_pick = QPushButton("Pick (P)")
        self.btn_reject = QPushButton("Reject (X)")
        self.btn_clear = QPushButton("Unflag (U)")
        self.btn_pick.clicked.connect(lambda: self.set_flag("pick"))
        self.btn_reject.clicked.connect(lambda: self.set_flag("reject"))
        self.btn_clear.clicked.connect(lambda: self.set_flag("none"))
        for b in (self.btn_pick, self.btn_reject, self.btn_clear):
            row.addWidget(b)
        sl.addLayout(row)
        stars = QHBoxLayout()
        for n in range(1, 6):
            b = QPushButton("★" * n if n < 3 else f"{n}★")
            b.setToolTip(f"Rate {n} star(s) - key {n}")
            b.clicked.connect(lambda _=False, n=n: self.set_rating(n))
            stars.addWidget(b)
        sl.addLayout(stars)
        self.btn_group = QPushButton("Show only this group (G)")
        self.btn_group.clicked.connect(self.toggle_group_filter)
        sl.addWidget(self.btn_group)

        split = QSplitter()
        split.addWidget(self.grid)
        split.addWidget(centre)
        split.addWidget(side)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([400, 760, 320])
        self.setCentralWidget(split)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.hide()
        self.status_lbl = QLabel("")
        self.statusBar().addWidget(self.status_lbl, 1)
        self.statusBar().addPermanentWidget(self.progress)

    def _build_actions(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_open = QAction("Open folder…", self, shortcut=QKeySequence.StandardKey.Open,
                                triggered=self.open_folder)
        self.act_analyze = QAction("Analyze", self, triggered=lambda: self.start_analysis(False))
        self.act_reanalyze = QAction("Re-analyze all", self, triggered=lambda: self.start_analysis(True))
        self.act_stop = QAction("Stop", self, triggered=self.stop_analysis)
        for a in (self.act_open, self.act_analyze, self.act_stop):
            tb.addAction(a)

        tb.addSeparator()
        tb.addWidget(QLabel(" Mode "))
        self.mode_box = QComboBox()
        self.mode_box.addItem("Fast (embedded preview)", "fast")
        self.mode_box.addItem("Accurate (decode RAW)", "accurate")
        self.mode_box.setToolTip("Fast reads the JPEG inside each ARW; Accurate demosaics the RAW (about 10x slower).")
        tb.addWidget(self.mode_box)

        tb.addWidget(QLabel("   Show "))
        self.filter_box = QComboBox()
        for text, key in (("All", "all"), ("Picks", "pick"), ("Rejected", "reject"), ("Unflagged", "none"),
                          ("4★ and up", "4"), ("Best of each group", "best")):
            self.filter_box.addItem(text, key)
        self.filter_box.currentIndexChanged.connect(lambda _: self.rebuild_grid())
        tb.addWidget(self.filter_box)

        tb.addWidget(QLabel("   Sort "))
        self.sort_box = QComboBox()
        for text, key in (("Capture order", "time"), ("Score (high to low)", "score"), ("Group", "group")):
            self.sort_box.addItem(text, key)
        self.sort_box.currentIndexChanged.connect(lambda _: self.rebuild_grid())
        tb.addWidget(self.sort_box)

        self.act_write = QAction("Write XMP ratings", self, triggered=self.write_xmp)
        self.act_export = QAction("Export picks…", self, triggered=self.export_picks_dialog)
        self.act_csv = QAction("Export CSV…", self, triggered=self.export_csv_dialog)
        self.act_lr = QAction("Also write Lightroom-style .xmp", self, checkable=True)
        self.act_lr.setChecked(bool(self.settings_store.value("lr_xmp", False, bool)))
        self.act_help = QAction("Keyboard shortcuts", self, triggered=self.show_help)

        menu = self.menuBar().addMenu("&File")
        menu.addAction(self.act_open)
        menu.addAction(self.act_reanalyze)
        menu.addSeparator()
        for a in (self.act_write, self.act_export, self.act_csv, self.act_lr):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(QAction("Quit", self, shortcut=QKeySequence.StandardKey.Quit, triggered=self.close))
        self.menuBar().addMenu("&Help").addAction(self.act_help)

        tb.addSeparator()
        for a in (self.act_write, self.act_export):
            tb.addAction(a)

        def sc(keys, fn):
            QShortcut(QKeySequence(keys), self, activated=fn)

        sc("P", lambda: self.set_flag("pick"))
        sc("X", lambda: self.set_flag("reject"))
        sc("U", lambda: self.set_flag("none"))
        for n in range(0, 6):
            sc(str(n), lambda n=n: self.set_rating(n))
        sc("]", lambda: self.jump_group(+1))
        sc("[", lambda: self.jump_group(-1))
        sc("F", self.fit_preview)
        sc("Z", self.toggle_100)
        sc("O", self.preview.toggle_overlays)
        sc("G", self.toggle_group_filter)

    # ---- state helpers ---------------------------------------------------------------
    def _update_state(self):
        busy = self.worker is not None and self.worker.isRunning()
        has = self.folder is not None
        self.act_analyze.setEnabled(has and not busy)
        self.act_reanalyze.setEnabled(has and not busy)
        self.act_stop.setEnabled(busy)
        self.act_open.setEnabled(not busy)
        for a in (self.act_write, self.act_export, self.act_csv):
            a.setEnabled(bool(self.results) and not busy)
        self.mode_box.setEnabled(not busy)

    def _msg(self, text: str):
        self.status_lbl.setText(text)

    # ---- folder / analysis ----------------------------------------------------------
    def open_folder(self, folder: str | None = None):
        if not folder:
            start = self.settings_store.value("last_folder", str(Path.home()))
            folder = QFileDialog.getExistingDirectory(self, "Choose a folder of photos", start)
        if not folder:
            return
        self.settings_store.setValue("last_folder", folder)
        self.folder = folder
        self.results, self.view_rows, self.current = [], [], None
        self.thumb_cache.clear()
        self.preview_cache.clear()
        self.group_filter = None
        self.grid.clear()
        self.preview.clear_image("Analysing…")
        self.setWindowTitle(f"RawCull - {folder}")
        self.start_analysis(False)

    def start_analysis(self, force: bool):
        if not self.folder or (self.worker and self.worker.isRunning()):
            return
        self.settings.mode = self.mode_box.currentData()
        self.settings.write_lightroom_xmp = self.act_lr.isChecked()
        self.worker = AnalysisThread(self.folder, self.settings, force, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_analysis_done)
        self.worker.failed.connect(self._on_analysis_failed)
        self.progress.setValue(0)
        self.progress.show()
        self._msg("Analysing…")
        self.worker.start()
        self._update_state()

    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel_event.set()
            self._msg("Stopping…")

    def _on_progress(self, done, total, name):
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self._msg(f"Analysing {done}/{total}  {name}")

    def _on_analysis_failed(self, text):
        self.progress.hide()
        self._update_state()
        QMessageBox.critical(self, "Analysis failed", text)

    def _on_analysis_done(self, results):
        self.progress.hide()
        self.results = results
        self.thumb_cache.clear()
        self.rebuild_grid()
        self._update_state()
        ok = [r for r in results if not r.get("error")]
        picks = sum(1 for r in ok if effective(r)[1] == "pick")
        rej = sum(1 for r in ok if effective(r)[1] == "reject")
        groups = len({r["group"] for r in ok})
        note = "" if exif_mod.have_exiftool() else "  ·  exiftool not found: using sharpest-area instead of the camera AF point"
        if not faces_available():
            note += "  ·  face detection off (install the opencv-data package)"
        self._msg(f"{len(ok)} photos · {groups} groups · {picks} picks · {rej} auto-rejected"
                  f" · {len(results) - len(ok)} unreadable{note}")
        if not ok:
            self.preview.clear_image("No readable photos found in this folder")

    # ---- grid ----------------------------------------------------------------------
    def _visible(self, r: dict) -> bool:
        if r.get("error"):
            return self.filter_box.currentData() == "all"
        if self.group_filter is not None and r["group"] != self.group_filter:
            return False
        key = self.filter_box.currentData()
        rating, flag = effective(r)
        return {"all": True, "pick": flag == "pick", "reject": flag == "reject",
                "none": flag == "none", "4": rating >= 4,
                "best": r["group_rank"] == 1}.get(key, True)

    def rebuild_grid(self):
        keep = self.current["path"] if self.current else None
        rows = [r for r in self.results if self._visible(r)]
        sort = self.sort_box.currentData()
        if sort == "score":
            rows.sort(key=lambda r: -r.get("score", 0))
        elif sort == "group":
            rows.sort(key=lambda r: (r.get("group", 0), r.get("group_rank", 0)))
        else:
            rows.sort(key=lambda r: (r.get("capture_ts", 0), r["name"]))
        self.view_rows = rows
        self.grid.blockSignals(True)
        self.grid.clear()
        for r in rows:
            it = QListWidgetItem(QIcon(self._icon_for(r)), "")
            it.setToolTip(f"{r['name']}\nscore {r.get('score', 0):.0f}")
            self.grid.addItem(it)
        self.grid.blockSignals(False)
        target = next((i for i, r in enumerate(rows) if r["path"] == keep), 0 if rows else -1)
        if target >= 0:
            self.grid.setCurrentRow(target)
            self._on_row_changed(target)
        else:
            self.current = None
            self._show_details(None)

    def _icon_for(self, r: dict) -> QPixmap:
        rating, flag = effective(r)
        canvas = QPixmap(ICON_W, ICON_H)
        canvas.fill(QColor("#0d1117"))
        p = QPainter(canvas)
        if r.get("thumb"):
            pm = self.thumb_cache.get(r["thumb"])
            if pm is None:
                pm = QPixmap(r["thumb"])
                self.thumb_cache[r["thumb"]] = pm
            pm = pm.scaled(ICON_W - 6, ICON_H - 22, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            p.drawPixmap((ICON_W - pm.width()) // 2, 3 + (ICON_H - 22 - pm.height()) // 2, pm)
        color = GREEN if flag == "pick" else RED if flag == "reject" else None
        if color:
            p.setPen(QPen(color, 4))
            p.drawRect(2, 2, ICON_W - 4, ICON_H - 4)
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        p.setPen(QColor("#e6edf3"))
        stars = "★" * max(rating, 0) if flag != "reject" else "✕"
        p.drawText(6, ICON_H - 6, f"{stars}")
        p.setPen(QColor("#8b949e"))
        p.drawText(ICON_W - 74, ICON_H - 6, f"G{r.get('group', -1)}  {r.get('score', 0):.0f}")
        if r.get("user"):
            p.setPen(AMBER)
            p.drawText(ICON_W - 14, 16, "●")
        p.end()
        return canvas

    def _refresh_item(self, r: dict):
        try:
            row = self.view_rows.index(r)
        except ValueError:
            return
        self.grid.item(row).setIcon(QIcon(self._icon_for(r)))

    # ---- selection / preview --------------------------------------------------------
    def _on_row_changed(self, row: int):
        if row < 0 or row >= len(self.view_rows):
            return
        r = self.view_rows[row]
        self.current = r
        self.zoom100 = False
        self._show_details(r)
        if r.get("error"):
            self.preview.clear_image(r["error"])
            return
        try:
            img = self.preview_cache.get(r["path"])
            if img is None:
                img = bgr_to_qimage(load_preview(r["path"], "fast").bgr)
                if len(self.preview_cache) > 24:
                    self.preview_cache.pop(next(iter(self.preview_cache)))
                self.preview_cache[r["path"]] = img
            self.preview.fit_mode = True
            self.preview.set_image(img)
            self.preview.set_overlays(r["roi"], r["faces"], r["roi_source"])
        except Exception as exc:
            self.preview.clear_image(f"Cannot show preview: {exc}")

    def _show_details(self, r: dict | None):
        if r is None:
            self.name_lbl.setText("-")
            for b in self.bars.values():
                b.setValue(0)
            self.score_lbl.setText("")
            self.state_lbl.setText("")
            self.exif_lbl.setText("")
            self.group_lbl.setText("")
            self.reason_lbl.setText("")
            return
        self.name_lbl.setText(r["name"])
        if r.get("error"):
            self.score_lbl.setText("-")
            self.reason_lbl.setText(r["error"])
            return
        rating, flag = effective(r)
        self.score_lbl.setText(f"{r['score']:.0f}<span style='font-size:12pt;color:#8b949e'> / 100</span>")
        manual = " (manual)" if r.get("user") else " (auto)"
        txt = {"pick": "<span style='color:#3fb950'>PICK</span>",
               "reject": "<span style='color:#f85149'>REJECT</span>", "none": "unflagged"}[flag]
        self.state_lbl.setText(f"{txt} · {'★' * rating if rating else 'no stars'}{manual}")
        self.exif_lbl.setText(exif_mod.summary(r.get("exif", {})))
        self.bars["focus"].setValue(int(r["focus_s"] * 100))
        self.bars["exposure"].setValue(int(r["exposure_s"] * 100))
        if r["face_s"] is None:
            self.bars["face"].setValue(0)
            self.bars["face"].setFormat("no face")
        else:
            self.bars["face"].setValue(int(r["face_s"] * 100))
            self.bars["face"].setFormat("%p%")
        src = {"af_point": "camera AF point", "face": "main face", "peak": "sharpest area"}[r["roi_source"]]
        self.group_lbl.setText(f"Group {r['group']} · rank {r['group_rank']} of {r['group_size']} · focus measured at {src}")
        self.reason_lbl.setText("<br>• " + "<br>• ".join(r["reasons"]) if r["reasons"] else "")

    # ---- actions ----------------------------------------------------------------------
    def _touch(self):
        r = self.current
        if r is None or r.get("error"):
            return None
        return r

    def set_flag(self, flag: str):
        r = self._touch()
        if r is None:
            return
        r.setdefault("user", {})
        if flag == "none" and "rating" not in r["user"]:
            r.pop("user", None)
        else:
            r["user"]["flag"] = flag
        self._after_change(r)

    def set_rating(self, n: int):
        r = self._touch()
        if r is None:
            return
        r.setdefault("user", {})["rating"] = n
        self._after_change(r)

    def _after_change(self, r: dict):
        save_decisions(self.folder, self.results)
        self._refresh_item(r)
        self._show_details(r)

    def jump_group(self, step: int):
        if not self.current or not self.view_rows:
            return
        cur = self.view_rows.index(self.current)
        gid = self.current["group"]
        rng = range(cur + 1, len(self.view_rows)) if step > 0 else range(cur - 1, -1, -1)
        for i in rng:
            if self.view_rows[i]["group"] != gid:
                # land on the best shot of that group when it is visible
                g = self.view_rows[i]["group"]
                best = next((j for j, r in enumerate(self.view_rows)
                             if r["group"] == g and r.get("group_rank") == 1), i)
                self.grid.setCurrentRow(best)
                return

    def toggle_group_filter(self):
        if self.group_filter is not None:
            self.group_filter = None
            self.btn_group.setText("Show only this group (G)")
        elif self.current and not self.current.get("error"):
            self.group_filter = self.current["group"]
            self.btn_group.setText("Show all photos (G)")
        self.rebuild_grid()

    def fit_preview(self):
        self.zoom100 = False
        self.preview.fit()

    def toggle_100(self):
        if not self.current or self.current.get("error"):
            return
        if self.zoom100:
            self.fit_preview()
            return
        self.zoom100 = True
        path = self.current["path"]
        self._msg("Loading full resolution…")
        self.full_thread = FullResThread(path, self)
        self.full_thread.done.connect(self._on_fullres)
        self.full_thread.failed.connect(lambda p, e: self._msg(f"Full-res failed: {e}"))
        self.full_thread.start()

    def _on_fullres(self, path, arr):
        if not self.current or self.current["path"] != path or not self.zoom100:
            return
        self.preview.set_image(rgb_to_qimage(arr))
        self.preview.set_overlays(self.current["roi"], self.current["faces"], self.current["roi_source"])
        self.preview.show_100()
        self._msg(f"{self.current['name']} at 100% - drag to pan, Z to go back")

    # ---- exports ----------------------------------------------------------------------
    def write_xmp(self):
        if QMessageBox.question(
                self, "Write XMP sidecars",
                "Write star ratings and pick/reject flags to XMP sidecar files next to your photos?\n\n"
                "RAW files are never modified. Existing sidecars are edited in place (a .rawcull.bak "
                "copy is kept) so darktable / Lightroom edits are preserved.") != QMessageBox.StandardButton.Yes:
            return
        self.settings_store.setValue("lr_xmp", self.act_lr.isChecked())
        n = write_xmp_all(self.results, self.act_lr.isChecked())
        self._msg(f"XMP sidecars written for {n} photos. In darktable use 'import' or 'load sidecar file'; "
                  f"in digiKam use 'Read metadata from files'.")

    def export_picks_dialog(self):
        dest = QFileDialog.getExistingDirectory(self, "Folder for picked photos")
        if not dest:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Export picks")
        box.setText("Link or copy the picked RAW files?")
        link = box.addButton("Symlinks (no extra disk space)", QMessageBox.ButtonRole.AcceptRole)
        copy = box.addButton("Copy files", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() not in (link, copy):
            return
        n = export_picks(self.results, dest, copy=box.clickedButton() is copy)
        self._msg(f"{n} picked photos exported to {dest}")

    def export_csv_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "rawcull_results.csv", "CSV (*.csv)")
        if path:
            export_csv([r for r in self.results if not r.get("error")], path)
            self._msg(f"CSV saved: {path}")

    def show_help(self):
        QMessageBox.information(self, "Keyboard shortcuts", (
            "Arrow keys   move between photos\n"
            "P / X / U    pick / reject / unflag\n"
            "0-5          star rating\n"
            "[  ]         previous / next group (jumps to its best shot)\n"
            "G            show only the current group (compare burst frames)\n"
            "Z            100% zoom on the focus box (loads the full RAW)\n"
            "F            fit to window     O  toggle overlays\n"
            "Wheel        zoom,  drag to pan,  double-click toggles fit / 100%"))


def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#0d1117"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#0d1117"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#161b22"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#21262d"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#1f6feb"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#161b22"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#e6edf3"))
    app.setPalette(pal)


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("RawCull")
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args and os.path.isdir(args[0]):
        win.open_folder(args[0])
    return app.exec()
