"""
Analysis Dashboard (PyQt6 + Plotly) - Interactive

Instructions:
1) Install dependencies:
   pip install pandas plotly scikit-learn PyQt6 PyQt6-WebEngine

2) Save this file into your project, e.g. app/software/analysis_dashboard.py
3) Import and instantiate AnalysisTab inside your main PyQt window and add it to the layout.

Notes:
- This module is deliberately self-contained and uses an on-disk cache folder
  (".analysis_cache") to speed repeated loads of many CSV files.
- It expects CSV files with the exact header you provided (columns):
  layout_id,run,step,robot,j_cmd_0,j_cmd_1,j_cmd_2,j_cmd_3,j_cmd_4,gripper_cmd,
  j_final_0,j_final_1,j_final_2,j_final_3,j_final_4,x_real,y_real,z_real,
  x_virtual,y_virtual,z_virtual,latency_ms,wait_time,elapsed_time,total_elapsed_time

Behavior:
- Left control panel: load folder, select layouts (or select "Últimos 5"), filter robot,
  set heatmap bins, toggle clustering.
- Right area: QTabWidget with 4 tabs (Temporal, Articular, Espacial, Clustering).
- Data loading is done on demand per layout; processed per-file pickles are stored
  in <csv_folder>/.analysis_cache/ for faster reload.

"""

import os
import re
import sys
import math
import pickle
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QLabel, QCheckBox, QSlider, QTabWidget,
    QComboBox, QSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objs as go
import plotly.subplots as sp
from app.software.ia_model.sequence_predictor import SequencePredictor


# -----------------------------
# Utility / Engine
# -----------------------------

CSV_HEADER = [
    'layout_id','run','step','robot',
    'j_cmd_0','j_cmd_1','j_cmd_2','j_cmd_3','j_cmd_4','gripper_cmd',
    'j_final_0','j_final_1','j_final_2','j_final_3','j_final_4',
    'x_real','y_real','z_real','x_virtual','y_virtual','z_virtual',
    'latency_ms','wait_time','elapsed_time','total_elapsed_time'
]

LAYOUT_FILENAME_RE = re.compile(r"([0-9a-fA-F]{6,})")  # tries to find hex id in filename

class AnalysisEngine:
    """Engine that scans folder, lazily loads CSVs and caches processed data."""

    def __init__(self, folder=None):
        self.folder = folder
        self.files = []
        self.file_mtimes = {}
        self.layout_to_files = {}
        self._memory_cache = {}

        self.seq_predictor = SequencePredictor()

        self.historical_scores = {}
        self._historical_array = None

        

    def scan_folder(self, folder):
        if not folder:
            return
        self.folder = folder
        cache_dir = os.path.join(folder, '.analysis_cache')
        os.makedirs(cache_dir, exist_ok=True)

        files = []
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.lower().endswith('.csv'):
                files.append(entry.path)
                self.file_mtimes[entry.path] = entry.stat().st_mtime

        # sort by mtime ascending
        files.sort(key=lambda p: self.file_mtimes.get(p, 0))
        self.files = files
        self.layout_to_files = {}

        # map layout ids by filename quick extract (faster than reading whole CSVs)
        for p in self.files:
            lid = self._extract_layout_id_from_filename(p)
            if lid is None:
                # try reading first line to get layout_id column
                try:
                    row = pd.read_csv(p, nrows=1)
                    lid = str(row['layout_id'].iloc[0]) if 'layout_id' in row.columns else os.path.basename(p)
                except Exception:
                    lid = os.path.basename(p)
            self.layout_to_files.setdefault(lid, []).append(p)

        return list(self.layout_to_files.keys())
    


    def detect_unstable_steps(self, csv_path):

        df = self.load_and_process_file(csv_path)

        if df is None or df.empty:
            return None

        jcols = ["j_cmd_0","j_cmd_1","j_cmd_2","j_cmd_3","j_cmd_4"]

        # delta articular
        delta = df[jcols].diff().abs()

        # jerk
        jerk = delta.diff().abs()

        df["jerk_score"] = jerk.mean(axis=1)

        # score combinado
        df["instability_score"] = (
            df["jerk_score"] * 0.6 +
            df["latency_ms"] * 0.2 +
            df["elapsed_time"] * 0.2
        )

        # top 10 pasos más inestables
        worst = df.nlargest(10, "instability_score")

        return worst[["step","instability_score","jerk_score","latency_ms","elapsed_time"]]
        

    def build_historical_index(self):

        cache_file = os.path.join(self.folder, ".analysis_cache", "historical_index.pkl")

        # --------------------------------
        # Si existe cache → cargarlo
        # --------------------------------
        if os.path.exists(cache_file):

            try:
                data = pd.read_pickle(cache_file)

                self.historical_scores = data["scores"]
                self._historical_array = np.array(list(self.historical_scores.values()))

                print("Historical index cargado desde cache")

                return

            except Exception:
                print("Cache corrupto, reconstruyendo...")

        # --------------------------------
        # Si no existe → construir
        # --------------------------------

        print("Construyendo índice histórico...")

        scores = {}

        for lid, files in self.layout_to_files.items():

            files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
            csv_path = files_sorted[-1]

            pred = self.seq_predictor.predict_from_csv(csv_path)

            if pred:
                scores[lid] = pred["dynamic_index"]

        self.historical_scores = scores
        self._historical_array = np.array(list(scores.values()))

        # --------------------------------
        # Guardar cache
        # --------------------------------

        os.makedirs(os.path.join(self.folder, ".analysis_cache"), exist_ok=True)

        pd.to_pickle(
            {"scores": scores},
            cache_file
        )

        print("Historical index guardado en cache")

    def compute_smoothness_score(self, csv_path):

        df = self.load_and_process_file(csv_path)

        if df is None or df.empty:
            return 0

        # si jerk no existe lo calculamos
        if "jerk_mean" not in df.columns:

            jcols = ["j_cmd_0","j_cmd_1","j_cmd_2","j_cmd_3","j_cmd_4"]

            delta = df[jcols].diff().abs()
            jerk = delta.diff().abs()

            jerk_mean = jerk.mean().mean()

        else:
            jerk_mean = df["jerk_mean"].mean()

        if np.isnan(jerk_mean):
            return 0

        normalized = min(jerk_mean / 40.0, 1.0)

        smooth_score = 100 * (1 - normalized)

        return max(0, min(100, smooth_score))



    def get_percentile(self, value):
        if not hasattr(self, "_historical_array"):
            return None
        return (self._historical_array < value).mean() * 100

    def _extract_layout_id_from_filename(self, path):
        name = os.path.basename(path)
        m = LAYOUT_FILENAME_RE.search(name)
        if m:
            return m.group(1)
        return None

    def get_all_layout_ids(self):
        return list(self.layout_to_files.keys())

    def get_most_recent_layouts(self, n=5):
        # Use file mtime to pick last N unique layout ids
        ordered = sorted(self.files, key=lambda p: os.path.getmtime(p))
        res = []
        seen = set()
        for p in reversed(ordered):
            lid = self._extract_layout_id_from_filename(p)
            if lid is None:
                try:
                    row = pd.read_csv(p, nrows=1)
                    lid = str(row['layout_id'].iloc[0])
                except Exception:
                    lid = os.path.basename(p)
            if lid not in seen:
                seen.add(lid)
                res.append(lid)
            if len(res) >= n:
                break
        return list(reversed(res))

    def _cache_path(self, csv_path):
        folder = os.path.dirname(csv_path)
        cache_dir = os.path.join(folder, '.analysis_cache')
        os.makedirs(cache_dir, exist_ok=True)
        fn = os.path.basename(csv_path) + '.pkl'
        return os.path.join(cache_dir, fn)

    def load_and_process_file(self, csv_path, force_reload=False):
        """Load a single CSV and compute derived features. Uses a disk cache per CSV to speed up future loads."""
        if not force_reload and csv_path in self._memory_cache:
            return self._memory_cache[csv_path]

        cache_file = self._cache_path(csv_path)
        csv_mtime = os.path.getmtime(csv_path)

        # if cache exists and newer than csv, load it
        if os.path.exists(cache_file):
            try:
                meta_path = cache_file + '.meta'
                if os.path.exists(meta_path):
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                    if meta.get('csv_mtime') == csv_mtime:
                        df_cached = pd.read_pickle(cache_file)
                        self._memory_cache[csv_path] = df_cached
                        return df_cached
            except Exception:
                pass

        # otherwise, read CSV and compute features
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            return None

        # ensure expected columns exist
        missing = [c for c in CSV_HEADER if c not in df.columns]
        if missing:
            print(f"Warning: file {csv_path} missing expected columns: {missing}")

        # Normalize types and names
        # Convert robot to string
        df['robot'] = df['robot'].astype(str)

        # Group by run to compute joint deltas per run sequence
        jcols = ['j_cmd_0','j_cmd_1','j_cmd_2','j_cmd_3','j_cmd_4']
        for jc in jcols:
            if jc not in df.columns:
                df[jc] = 0.0

        # Ensure step and run are sorted
        if 'step' in df.columns:
            df = df.sort_values(['run','step']).reset_index(drop=True)

        # compute deltas within each run
        df[['delta_j0','delta_j1','delta_j2','delta_j3','delta_j4']] = df.groupby('run')[jcols].diff().abs().fillna(0.0)
        df['delta_j_max'] = df[['delta_j0','delta_j1','delta_j2','delta_j3','delta_j4']].max(axis=1)
        df['delta_j_sum'] = df[['delta_j0','delta_j1','delta_j2','delta_j3','delta_j4']].sum(axis=1)

        # -----------------------------
        # JERK (segunda derivada)
        # -----------------------------
        df[['jerk_j0','jerk_j1','jerk_j2','jerk_j3','jerk_j4']] = (
            df.groupby('run')[['delta_j0','delta_j1','delta_j2','delta_j3','delta_j4']]
            .diff()
            .abs()
            .fillna(0.0)
        )

        df['jerk_max'] = df[['jerk_j0','jerk_j1','jerk_j2','jerk_j3','jerk_j4']].max(axis=1)
        df['jerk_mean'] = df[['jerk_j0','jerk_j1','jerk_j2','jerk_j3','jerk_j4']].mean(axis=1)

        # error between real and virtual
        df['error_xyz'] = np.sqrt(
            (df['x_real'] - df['x_virtual'])**2 +
            (df['y_real'] - df['y_virtual'])**2 +
            (df['z_real'] - df['z_virtual'])**2
        )

        # ensure numeric conversions
        df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce').fillna(0.0)
        df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce').fillna(0.0)
        df['wait_time'] = pd.to_numeric(df['wait_time'], errors='coerce').fillna(0.0)

        # tag layout id (if file doesn't include it reliably, extract from filename)
        if 'layout_id' not in df.columns or df['layout_id'].isna().all():
            lid = self._extract_layout_id_from_filename(csv_path) or os.path.basename(csv_path)
            df['layout_id'] = lid
            

        # store cache on disk
        try:
            pd.to_pickle(df, cache_file)
            with open(cache_file + '.meta', 'wb') as f:
                pickle.dump({'csv_mtime': csv_mtime, 'processed_at': datetime.utcnow()}, f)
        except Exception as e:
            print(f"Warning: could not write cache for {csv_path}: {e}")

        self._memory_cache[csv_path] = df
        return df

    def load_selected_layouts(self, layout_ids, robot_filter='Both'):
        """Return concatenated DataFrame for selected layout_ids and robot filter.
        layout_ids: list of layout ids
        robot_filter: 'Both', 'Robot 1', 'Robot 2' or values appearing in csv 'Robot 1' etc.
        """
        if not self.folder:
            return pd.DataFrame()

        to_concat = []
        # find files for each layout id (use latest file for that layout)
        for lid in layout_ids:
            files = self.layout_to_files.get(lid, [])
            if not files:
                continue
            # choose most recent file for that layout
            files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
            chosen = files_sorted[-1]
            df = self.load_and_process_file(chosen)
            if df is None:
                continue
            to_concat.append(df)

        if not to_concat:
            return pd.DataFrame()

        big = pd.concat(to_concat, ignore_index=True)

        if robot_filter == 'Robot 1':
            big = big[big['robot'].str.contains('1')]
        elif robot_filter == 'Robot 2':
            big = big[big['robot'].str.contains('2')]

        return big

# -----------------------------
# Plotly Tab Widgets
# -----------------------------

class BasePlotTab(QWidget):
    def __init__(self):
        super().__init__()
        self._web = None
        # crea el layout vacío (el QWebEngineView se añadirá después)
        layout = QVBoxLayout(self)
        # placeholder HTML inicial
        self._placeholder = QLabel("No data yet. Load folder and select layouts.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._placeholder)

    def _ensure_web(self):
        if self._web is None:
            try:
                self._web = QWebEngineView()
            except Exception as e:
                # si por alguna razón falla la creación del webengine, mostramos mensaje
                self._placeholder.setText(f"Error creating QWebEngineView: {e}")
                return False
            # quitar placeholder y añadir web view
            layout = self.layout()
            layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            layout.addWidget(self._web)
        return True

    def set_html_fig(self, fig):
        if not self._ensure_web():
            return
        html = fig.to_html(include_plotlyjs='cdn')
        # setHtml es seguro siempre que webview exista y QApplication esté activo
        self._web.setHtml(html)

    def clear(self):
        if self._web:
            self._web.setHtml('<html><body><h3>No data</h3></body></html>')
        else:
            # si no existe web todavía, actualizamos placeholder
            self._placeholder.setText('No data')


class TemporalTab(BasePlotTab):
    def update_data(self, df, selected_layouts):
        if df is None or df.empty:
            self.clear(); return

        fig = sp.make_subplots(rows=2, cols=1, subplot_titles=("Tiempo acumulado vs Step","Histograma elapsed_time"), row_heights=[0.6,0.4])

        # Tiempo acumulado vs step (group by layout and run)
        for lid in (selected_layouts or df['layout_id'].unique()):
            dfl = df[df['layout_id'] == lid]
            # average cumulative_time per step across runs
            # 🔥 Tiempo acumulado vs step usando total_elapsed_time

            if 'total_elapsed_time' in dfl.columns:

                avg = dfl.groupby('step')['total_elapsed_time'].mean().reset_index()

                fig.add_trace(
                    go.Scatter(
                        x=avg['step'],
                        y=avg['total_elapsed_time'],
                        mode='lines',
                        name=str(lid)
                    ),
                    row=1,
                    col=1
                )

        # Histograma
        fig.add_trace(go.Histogram(x=df['elapsed_time'], nbinsx=60, name='elapsed_time'), row=2, col=1)

        fig.update_layout(height=800, showlegend=True)
        self.set_html_fig(fig)


class JointTab(BasePlotTab):
    def update_data(self, df, selected_layouts):
        if df is None or df.empty:
            self.clear(); return

        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("ΔJ_max vs elapsed_time","Boxplot ΔJ per joint","Latency vs elapsed","ΔJ distribution"))

        # Scatter ΔJ_max vs elapsed
        fig.add_trace(go.Scatter(x=df['delta_j_max'], y=df['elapsed_time'], mode='markers', marker={'size':4}, name='ΔJmax'), row=1, col=1)

        # Boxplot ΔJ per joint
        for i, col in enumerate(['delta_j0','delta_j1','delta_j2','delta_j3','delta_j4']):
            fig.add_trace(go.Box(y=df[col], name=col), row=1, col=2)

        # Latency vs elapsed
        fig.add_trace(go.Scatter(x=df['latency_ms'], y=df['elapsed_time'], mode='markers', marker={'size':4}, name='latency'), row=2, col=1)

        # ΔJ distribution
        fig.add_trace(go.Histogram(x=df['delta_j_max'], nbinsx=50, name='ΔJmax dist'), row=2, col=2)

        fig.update_layout(height=800, showlegend=False)
        self.set_html_fig(fig)


class SpatialTab(BasePlotTab):

    def update_data(self, df, bins=20):

        if df is None or df.empty:
            self.clear()
            return

        # 🔥 usar coordenadas virtuales
        df_clean = df.dropna(subset=["x_virtual", "y_virtual", "elapsed_time"])

        if df_clean.empty:
            self.clear()
            return

        x = df_clean["x_virtual"].values
        y = df_clean["y_virtual"].values
        z = df_clean["elapsed_time"].values

        try:
            counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
            sums, _, _ = np.histogram2d(
                x, y,
                bins=[xedges, yedges],
                weights=z
            )
        except Exception:
            self.clear()
            return

        avg = np.zeros_like(sums)
        mask = counts > 0
        avg[mask] = sums[mask] / counts[mask]

        heatmap = go.Heatmap(
            z=avg.T,
            x=(xedges[:-1] + xedges[1:]) / 2.0,
            y=(yedges[:-1] + yedges[1:]) / 2.0,
            colorscale="Hot",
            colorbar=dict(title="avg elapsed (s)")
        )

        fig = go.Figure(data=[heatmap])
        fig.update_layout(
            title="Heatmap XY (Virtual Workspace)",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            height=700
        )

        self.set_html_fig(fig)

class ClusteringTab(BasePlotTab):
    def __init__(self):
        super().__init__()
        self.model = None

    def run_kmeans(self, df, n_clusters=3):
        features = ['delta_j0','delta_j1','delta_j2','delta_j3','delta_j4','latency_ms','wait_time']
        X = df[features].fillna(0.0).values
        km = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        df['cluster'] = km.labels_
        self.model = km
        return df

    def update_data(self, df, n_clusters=3):
        if df is None or df.empty:
            self.clear(); return

        dfc = self.run_kmeans(df, n_clusters=n_clusters)

        fig = sp.make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=dfc['delta_j_max'], y=dfc['elapsed_time'], mode='markers', marker=dict(size=5, color=dfc['cluster'], showscale=True), text=dfc['layout_id']), row=1, col=1)
        fig.update_layout(title=f'KMeans clusters (k={n_clusters})', xaxis_title='ΔJ_max', yaxis_title='elapsed_time', height=700)
        self.set_html_fig(fig)

class AIInsightTab(BasePlotTab):

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def update_data(self, layout_ids):

        if not layout_ids:
            self.clear()
            return

        results = []

        for lid in layout_ids:
            files = self.engine.layout_to_files.get(lid, [])
            if not files:
                continue

            files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
            csv_path = files_sorted[-1]

            pred = self.engine.seq_predictor.predict_from_csv(csv_path)
            if pred is None:
                continue

            dynamic_index = pred["dynamic_index"]
            ineff_risk = pred["ineff_risk"]
            smooth_score = self.engine.compute_smoothness_score(csv_path)

            # Híbrido
            hybrid_index = (
                0.6 * dynamic_index +
                0.4 * smooth_score
            )

            results.append((lid, hybrid_index, ineff_risk, smooth_score))

        if not results:
            self.clear()
            return

        latest = results[-1]

        # -----------------------------
        # Detectar pasos problemáticos
        # -----------------------------

        files = self.engine.layout_to_files.get(latest[0], [])
        files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
        csv_path = files_sorted[-1]

        unstable_steps = self.engine.detect_unstable_steps(csv_path)
        hybrid_index = latest[1]
        ineff_risk = latest[2]
        smooth_score = latest[3]

        # Percentil histórico
        percentile = self.engine.get_percentile(hybrid_index)
        if percentile is None:
            percentile = 0

        fig = sp.make_subplots(
            rows=3,
            cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "xy"}],
                [{"type": "xy"}, None],
            ],
            subplot_titles=(
                "Dynamic Stability (Hybrid)",
                "Inefficiency Risk",
                "Smoothness Score",
                "Comparación layouts",
                "Top unstable steps",
                ""
            )
)
        # Hybrid Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=hybrid_index,
            number={'suffix': " /100"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 60], 'color': "red"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"},
                ],
            }
        ), row=1, col=1)

        # Inefficiency
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=ineff_risk,
            number={'suffix': " %"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"},
                ],
            }
        ), row=1, col=2)

        # Smoothness
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=smooth_score,
            number={'suffix': " /100"},
            gauge={'axis': {'range': [0, 100]}}
        ), row=2, col=1)

        # Comparación barras
        lids = [r[0] for r in results]
        scores = [r[1] for r in results]

        fig.add_trace(go.Bar(
            x=lids,
            y=scores,
            name="Hybrid Index"
        ), row=2, col=2)

        fig.update_layout(height=950, showlegend=False)

        self.set_html_fig(fig)

        # -----------------------------
        # Gráfica pasos inestables
        # -----------------------------

        if unstable_steps is not None:

            fig.add_trace(
                go.Bar(
                    x=unstable_steps["step"],
                    y=unstable_steps["instability_score"],
                    name="Unstable steps"
                ),
                row=3,
                col=1
            )

            fig.update_xaxes(title_text="Step", row=3, col=1)
            fig.update_yaxes(title_text="Instability score", row=3, col=1)





class AnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.engine = AnalysisEngine()
        self.selected_layouts = []
        self.robot_filter = 'Both'
        self.bins = 20
        self.n_clusters = 3

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left control panel
        ctrl_layout = QVBoxLayout()

        self.btn_load = QPushButton('Cargar carpeta CSV')
        self.btn_load.clicked.connect(self.on_load_folder)
        ctrl_layout.addWidget(self.btn_load)

        ctrl_layout.addWidget(QLabel('Layouts detectados'))
        self.list_layouts = QListWidget()
        self.list_layouts.itemChanged.connect(self.on_layout_selection_changed)
        ctrl_layout.addWidget(self.list_layouts)

        self.chk_last5 = QCheckBox('Últimos 5')
        self.chk_last5.stateChanged.connect(self.on_toggle_last5)
        ctrl_layout.addWidget(self.chk_last5)

        ctrl_layout.addWidget(QLabel('Filtro Robot'))
        self.combo_robot = QComboBox()
        self.combo_robot.addItems(['Both','Robot 1','Robot 2'])
        self.combo_robot.currentTextChanged.connect(self.on_robot_filter_changed)
        ctrl_layout.addWidget(self.combo_robot)

        ctrl_layout.addWidget(QLabel('Bins heatmap'))
        self.slider_bins = QSlider(Qt.Orientation.Horizontal)
        self.slider_bins.setMinimum(10)
        self.slider_bins.setMaximum(60)
        self.slider_bins.setValue(self.bins)
        self.slider_bins.valueChanged.connect(self.on_bins_changed)
        ctrl_layout.addWidget(self.slider_bins)

        ctrl_layout.addWidget(QLabel('Clusters (k)'))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2,10)
        self.spin_k.setValue(self.n_clusters)
        self.spin_k.valueChanged.connect(self.on_k_changed)
        ctrl_layout.addWidget(self.spin_k)

        self.btn_refresh = QPushButton('Refrescar vistas')
        self.btn_refresh.clicked.connect(self.refresh_all_tabs)
        ctrl_layout.addWidget(self.btn_refresh)

        ctrl_layout.addStretch()

        # ---------------- RIGHT SIDE ----------------
        tabs_layout = QVBoxLayout()
        self.tabs = QTabWidget()

        self.temporal_tab = TemporalTab()
        self.joint_tab = JointTab()
        self.spatial_tab = SpatialTab()
        self.cluster_tab = ClusteringTab()
        self.ai_tab = AIInsightTab(self.engine)   # 🔥 AHORA sí

        self.tabs.addTab(self.temporal_tab, 'Temporal')
        self.tabs.addTab(self.joint_tab, 'Articular')
        self.tabs.addTab(self.spatial_tab, 'Espacial')
        self.tabs.addTab(self.cluster_tab, 'Clustering')
        self.tabs.addTab(self.ai_tab, 'AI Insight')  # 🔥 aquí correcto

        tabs_layout.addWidget(self.tabs)

        main_layout.addLayout(ctrl_layout, 1)
        main_layout.addLayout(tabs_layout, 4)

    # ------------------ control callbacks ------------------
    def on_load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Seleccionar carpeta CSV')
        if not folder:
            return
        layout_ids = self.engine.scan_folder(folder)
        print("Construyendo índice histórico...")
        self.engine.build_historical_index()
        print("Índice histórico listo.")
        self.populate_layout_list(layout_ids)
        # select last 5 automatically
        last5 = self.engine.get_most_recent_layouts(5)
        self.select_layouts(last5)
        # auto refresh first view
        QTimer.singleShot(100, self.refresh_all_tabs)

    def populate_layout_list(self, layout_ids):
        self.list_layouts.clear()
        for lid in layout_ids:
            item = QListWidgetItem(str(lid))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.list_layouts.addItem(item)

    def select_layouts(self, layout_ids):
        # check matching items in list
        for i in range(self.list_layouts.count()):
            it = self.list_layouts.item(i)
            if it.text() in layout_ids:
                it.setCheckState(Qt.CheckState.Checked)

    def on_layout_selection_changed(self, item):
        # build selected list
        self.selected_layouts = [self.list_layouts.item(i).text() for i in range(self.list_layouts.count()) if self.list_layouts.item(i).checkState() == Qt.CheckState.Checked]
        # if there are selections, uncheck last5
        if self.selected_layouts:
            self.chk_last5.setChecked(False)
        QTimer.singleShot(50, self.refresh_all_tabs)

    def on_toggle_last5(self, state):
        if state == Qt.CheckState.Checked:
            last5 = self.engine.get_most_recent_layouts(5)
            self.select_layouts(last5)
        else:
            # leave selection as is
            pass
        QTimer.singleShot(50, self.refresh_all_tabs)

    def on_robot_filter_changed(self, txt):
        self.robot_filter = txt
        QTimer.singleShot(50, self.refresh_all_tabs)

    def on_bins_changed(self, val):
        self.bins = val
        QTimer.singleShot(50, self.refresh_all_tabs)

    def on_k_changed(self, val):
        self.n_clusters = val
        QTimer.singleShot(50, self.refresh_all_tabs)

    def on_layout_list_doubleclick(self, item):
        # optionally implement quick preview
        pass

    # ------------------ refresh / plotting ------------------
    def refresh_all_tabs(self):
        if not self.engine.folder:
            return

        # if no explicit selection, use last 5
        selected = self.selected_layouts or self.engine.get_most_recent_layouts(5)

        # load data (this will read and cache per-file)
        df = self.engine.load_selected_layouts(selected, robot_filter=self.robot_filter)

        # send to tabs
        # Temporal: needs selected list to overlay
        self.temporal_tab.update_data(df, selected)
        self.joint_tab.update_data(df, selected)
        self.spatial_tab.update_data(df, bins=self.bins)
        self.cluster_tab.update_data(df, n_clusters=self.n_clusters)
        # AI Insight (usa solo layout IDs)
        self.ai_tab.update_data(selected)

# -----------------------------
# If run standalone for testing
# -----------------------------
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = AnalysisTab()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec())
