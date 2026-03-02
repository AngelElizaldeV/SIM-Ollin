# trajectory_generator.py
import math
import numpy as np

def global_to_local(global_xy, robot_origin):
    """
    Convierte coordenadas globales XY (m) a coordenadas locales del robot (m).
    """
    return (
        global_xy[0] - robot_origin[0],
        global_xy[1] - robot_origin[1],
    )


def generate_robot_sequence_industrial(
        layout,          # [(x,y), ...] en metros
        params,          # dict con keys: robot1_stations, robot2_stations, tiempos_estacion, R1, R2, ...
        robot_id,        # "R1" o "R2"
        ik_model,        # instancia de RobotKinematics (usa xyz_to_joint)
        ps_z_mm=234.635322,   # altura PS fija (mm)
        pick_depth_mm=140.0   # profundidad de pick para estaciones (mm)
):
    sequence = []

    # constantes
    FEEDER_DEPTH_MM = 110  # 7 cm para feeders
    GRIPPER_OPEN = 1
    GRIPPER_CLOSED = 0
    PREP_JOINTS = [0.0, 58.0, -90.0, -53.0, 0.0]   # pose de preparación final
    # seleccionar base desde params si existe (en metros)
    base = params.get("R1") if robot_id == "R1" else params.get("R2")
    if base and isinstance(base, (list, tuple)) and len(base) >= 2:
        base_x, base_y = base[0], base[1]
    else:
        if robot_id == "R1":
            base_x, base_y = 0.5325, 0.65
        else:
            base_x, base_y = 0.5325, 1.2

    robot_name = "Robot 1" if robot_id == "R1" else "Robot 2"
    indices = params.get("robot1_stations") if robot_id == "R1" else params.get("robot2_stations")
    tiempos = params.get("tiempos_estacion", [0]*len(layout))

    def add_step(joints_arr, gripper, wait, label):
        if joints_arr is None:
            return
        try:
            joints_list = [float(round(float(j), 3)) for j in list(joints_arr)]
        except Exception:
            return
        sequence.append({
            "robot": robot_name,
            "joints": joints_list,
            "gripper": int(gripper),
            "wait": float(wait),
            "label": label
        })

    # Inicio: punto de preparación
    add_step(PREP_JOINTS, GRIPPER_OPEN, 0.0, f"Punto de Preparacion {robot_name}")

    current_gripper = GRIPPER_OPEN
    skip_next = False

    # recorrer estaciones en el orden asignado
    for pos_in_list, idx in enumerate(indices):
        if skip_next:
            skip_next = False
            print(f"Saltando indice {idx} porque fue atendido por transferencia previa")
            continue

        if idx < 0 or idx >= len(layout):
            print(f"Índice {idx} fuera del layout (len={len(layout)}) - se salta")
            continue

        xg, yg = layout[idx]   # metros globales
        xr = xg - base_x
        yr = yg - base_y
        xr_mm = xr * 1000.0
        yr_mm = yr * 1000.0

        # ----------------------------------------
# Calcular J0 estimado (ángulo base)
# ----------------------------------------
        j0_est = math.degrees(math.atan2(yr_mm, xr_mm))


        is_feeder = (idx == 0 or idx == len(layout)-1)

        # Orientación GLOBAL deseada en PyBullet
        # ----------------------------------------
        if is_feeder:
            global_orientation = 90      # paralelo eje X global
        else:
            global_orientation = 0     # paralelo eje Y global

        # Convertir orientación global a beta local del robot
        beta_orientation = global_orientation - j0_est



        # ---------------- PS (punto seguro)
        print(f"\nEstación {idx} - PS")
        print("XR_mm:", xr_mm, "YR_mm:", yr_mm, "Z PS:", ps_z_mm)

        res_ps = ik_model.xyz_to_joint(
    [xr_mm, yr_mm, ps_z_mm, -95, beta_orientation]
)
        if not res_ps or "joint" not in res_ps or res_ps.get("status", 1) != 0:
            print(f"❌ IK falló en PS estación {idx} -> resultado: {res_ps}")
            continue

        joints_ps = res_ps["joint"]

        # Evitar PS automático en primera estación de R2
        if not (robot_id == "R2" and pos_in_list == 0):
            add_step(joints_ps, current_gripper, 0.0, f"PS Est{idx}")


        # ---------------- PR (bajada)
        z_pr = ps_z_mm - (FEEDER_DEPTH_MM if is_feeder else pick_depth_mm)
        print(f"Estación {idx} - PR (bajar) Z_PR:", z_pr)

        res_pr = ik_model.xyz_to_joint(
                                        [xr_mm, yr_mm, z_pr, -95, beta_orientation]
                                    )

        if not res_pr or "joint" not in res_pr or res_pr.get("status", 1) != 0:
            print(f"❌ IK falló en PR estación {idx} -> resultado: {res_pr}")
            continue

        joints_pr = res_pr["joint"]

        # Evitar PR automático en primera estación de R2
        if not (robot_id == "R2" and pos_in_list == 0):
            add_step(joints_pr, current_gripper, 0.0, f"PR Est{idx}")


        # ----- CASO ESPECIAL: Primera estación de R2
        if robot_id == "R2" and pos_in_list == 0:

            # Ya estamos en PS porque se agregó arriba

            # 1️⃣ Esperar en PS
            wait_time = tiempos[idx] if idx < len(tiempos) else 0.0
            add_step(joints_ps, current_gripper, wait_time,
                    f"Espera inicial R2 Est{idx}")

            # 2️⃣ Bajar
            add_step(joints_pr, current_gripper, 0.0,
                    f"PR Est{idx}")

            # 3️⃣ Tomar
            current_gripper = GRIPPER_CLOSED
            add_step(joints_pr, current_gripper, 0.0,
                    f"Tomar Cubo Est{idx}")

            # 4️⃣ Subir
            add_step(joints_ps, current_gripper, 0.0,
                    f"PS Est{idx} (post-tomar)")

            continue

            # en caso de fallback, seguimos con la lógica normal (sin hacer skip)

                # ---------------- Lógica feeder vs estación (pick/place) - flujo normal
        if is_feeder:

            # ----- CASO ESPECIAL: Última estación de R2
            if robot_id == "R2" and pos_in_list == len(indices) - 1:

                # 1️⃣ Soltar
                current_gripper = GRIPPER_OPEN
                add_step(joints_pr, current_gripper, 0.0,
                        f"Soltar Cubo Final Est{idx}")

                # 2️⃣ Subir
                add_step(joints_ps, current_gripper, 0.0,
                        f"PS Final Est{idx}")

                # 3️⃣ Regresar a HOME
                add_step(PREP_JOINTS, current_gripper, 0.0,
                        f"Fin ciclo R2")

                break

            # ----- FEEDER NORMAL (tomar pieza)
            else:
                current_gripper = GRIPPER_CLOSED
                add_step(joints_pr, current_gripper, 0.0,
                        f"Tomar en feeder {idx}")
                add_step(joints_ps, current_gripper, 0.0,
                        f"PS after pick feeder {idx}")

        else:

            # ----- CASO ESPECIAL: Última estación de R1
            if robot_id == "R1" and pos_in_list == len(indices) - 1:

                # 1️⃣ Soltar pieza
                current_gripper = GRIPPER_OPEN
                add_step(joints_pr, current_gripper, 0.0,
                        f"Soltar Cubo Final Est{idx}")

                # 2️⃣ Subir a PS
                add_step(joints_ps, current_gripper, 0.0,
                        f"PS Est{idx} Final")

                # 3️⃣ Regresar a preparación abierto
                add_step(PREP_JOINTS, current_gripper, 0.0,
                        f"Fin ciclo R1")

                break

            # ----- CASO ESPECIAL: Última estación de R2 (si NO es feeder)
            if robot_id == "R2" and pos_in_list == len(indices) - 1:

                # 1️⃣ Soltar
                current_gripper = GRIPPER_OPEN
                add_step(joints_pr, current_gripper, 0.0,
                        f"Soltar Cubo Final Est{idx}")

                # 2️⃣ Subir
                add_step(joints_ps, current_gripper, 0.0,
                        f"PS Final Est{idx}")

                # 3️⃣ HOME
                add_step(PREP_JOINTS, current_gripper, 0.0,
                        f"Fin ciclo R2")

                break

            # ----- FLUJO NORMAL
            else:

                # 1️⃣ Soltar pieza
                current_gripper = GRIPPER_OPEN
                add_step(joints_pr, current_gripper, 0.0,
                        f"Soltar Cubo Est{idx}")

                # 2️⃣ Subir a PS con espera
                wait_time = tiempos[idx] if idx < len(tiempos) else 0.0
                add_step(joints_ps, current_gripper, wait_time,
                        f"PS Est{idx}")

                # 3️⃣ Bajar nuevamente
                add_step(joints_pr, current_gripper, 0.0,
                        f"PR Est{idx} (pre-tomar)")

                # 4️⃣ Tomar pieza procesada
                current_gripper = GRIPPER_CLOSED
                add_step(joints_pr, current_gripper, 0.0,
                        f"Tomar Cubo Est{idx}")

                # 5️⃣ Subir con pieza
                add_step(joints_ps, current_gripper, 0.0,
                        f"PS Est{idx} (post-tomar)")



    # Al final volver a punto de preparación (pose segura)
    add_step(PREP_JOINTS, current_gripper, 0.0, f"Punto de Preparacion {robot_name} (final)")
    return sequence
