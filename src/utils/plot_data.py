from pathlib import Path
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def animate_solution(
    X,
    T,
    U,
    title="Evolución de u(t, x)",
    save_path="animacion.gif",
    duration_seconds=8.0,
    target_fps=25,  # FPS estándar y compatible con cualquier visor de GIF
):
    """Anima la evolución temporal 1D de cualquier solución u(t, x) garantizando

    una duración exacta en segundos.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds debe ser mayor que cero")

    # -----------------------------------------------------------------
    # 1. Normalización de dimensiones
    # -----------------------------------------------------------------
    x = X[0, :] if X.ndim == 2 else X
    t = T[:, 0] if (T.ndim == 2 and T.shape[0] > 1) else T
    if T.ndim == 2 and T.shape[0] == 1:
        t = T[0, :]

    if U.shape == (len(x), len(t)):
        U = U.T

    Nt_orig, Nx = U.shape

    # -----------------------------------------------------------------
    # 2. Ajuste de Fotogramas (Subsampling para duración exacta)
    # -----------------------------------------------------------------
    # Calcular cuántos cuadros necesitamos en total (ej: 4s * 25fps = 100 frames)
    total_frames = int(round(duration_seconds * target_fps))

    if Nt_orig >= total_frames:
        # Seleccionamos exactamente 'total_frames' distribuidos uniformemente
        frame_indices = np.linspace(0, Nt_orig - 1, total_frames, dtype=int)
    else:
        # Si Nt_orig es menor que los frames deseados, usamos todos los disponibles
        frame_indices = np.arange(Nt_orig)

    # Filtrar las matrices con los índices seleccionados
    U = U[frame_indices, :]
    t = t[frame_indices]
    Nt = len(t)

    # Calcular el FPS y el intervalo exacto según el número final de frames
    fps = Nt / duration_seconds
    interval_ms = 1000.0 / fps

    # -----------------------------------------------------------------
    # 3. Configuración del gráfico
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_xlim(x.min(), x.max())
    u_min, u_max = U.min(), U.max()
    margin = 0.1 * (u_max - u_min) if u_max != u_min else 0.1
    ax.set_ylim(u_min - margin, u_max + margin)

    (line,) = ax.plot([], [], "b-", lw=2, label="u(t, x)")
    time_text = ax.text(
        0.02, 0.93, "", transform=ax.transAxes, fontsize=11, fontweight="bold"
    )

    ax.set_title(title)
    ax.set_xlabel("Espacio (x)")
    ax.set_ylabel("Velocidad (u)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    # -----------------------------------------------------------------
    # 4. Renderizado y Guardado
    # -----------------------------------------------------------------
    def update(frame):
        line.set_data(x, U[frame, :])
        time_text.set_text(f"Tiempo t = {t[frame]:.3f} s")
        return line, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=Nt, interval=interval_ms, blit=True
    )

    print(
        f"Generando animación de {duration_seconds}s ({Nt} frames a {fps:.1f} FPS)..."
    )
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close()
    print(f"¡Animación guardada con éxito en '{save_path}'!")