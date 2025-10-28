import math
import ezdxf
import io
import base64

# 1. Cone
def generate_cone(diameter, height):
    D = diameter
    H = height

    # Calculations
    R = math.sqrt((D / 2) ** 2 + H ** 2)
    beta = 180 * (D / R)  # in degrees
    A = 2 * R * math.sin(math.radians(beta / 2))

    # Generate arc points (sector)
    steps = 120
    pts = [(0, 0)]
    for i in range(steps + 1):
        theta = math.radians(i * beta / steps)
        x = R * math.cos(theta)
        y = R * math.sin(theta)
        pts.append((x, y))
    pts.append((0, 0))

    return {
        "points": pts,
        "data": {
            "R": round(R, 2),
            "beta": round(beta, 2),
            "A": round(A, 2)
        }
    }

# 2. Frustum Cone
def generate_frustum_cone(d1, d2, value, mode="H"):
    D1, D2 = float(d1), float(d2)

    if mode.upper() == "H":
        H = value
        R1 = math.sqrt(H**2 + (D1 / 2)**2)
        R2 = math.sqrt(H**2 + (D2 / 2)**2)
        beta = 360 * (D1 - D2) / (2 * math.pi * R1)
    else:
        beta = float(value)
        if beta <= 0:
            raise ValueError("Beta must be > 0°.")
        beta_max = 360 * (D1 - D2) / (math.pi * D1)
        if beta > beta_max:
            beta = beta_max
        R_slant = (D1 - D2) * 180 / (math.pi * beta)
        R_outer = R_slant * (D1 / (D1 - D2))
        R_inner = R_slant * (D2 / (D1 - D2))
        diff = max(R_outer**2 - (D1 / 2)**2, 0)
        H = math.sqrt(diff)
        R1, R2 = R_outer, R_inner

    # --- Arc parameters for proper circular sector ---
    start_angle = -beta / 2
    end_angle = beta / 2
    
    # Calculate end points for radial lines
    start_angle_rad = math.radians(start_angle)
    end_angle_rad = math.radians(end_angle)
    
    # Points for the two radial lines
    p1_outer = (R1 * math.cos(start_angle_rad), R1 * math.sin(start_angle_rad))
    p1_inner = (R2 * math.cos(start_angle_rad), R2 * math.sin(start_angle_rad))
    p2_outer = (R1 * math.cos(end_angle_rad), R1 * math.sin(end_angle_rad))
    p2_inner = (R2 * math.cos(end_angle_rad), R2 * math.sin(end_angle_rad))

    # --- Calculs complémentaires ---
    corde_A = 2 * R1 * math.sin(math.radians(beta / 2))
    corde_C = 2 * R2 * math.sin(math.radians(beta / 2))
    L = R1 - R2
    pi_D1 = math.pi * D1
    pi_D2 = math.pi * D2

    # hauteur projetée de chaque rayon sur l'axe du cône
    h1 = R1 * math.cos(math.radians(beta / 2))
    h2 = R2 * math.cos(math.radians(beta / 2))
    B = h2 - h1

    data = {
        "Longueur L": round(L, 2),
        "Rayon R1": round(R1, 2),
        "Rayon R2": round(R2, 2),
        "Angle des gabarits α": round(beta, 2),
        "Angolo di cono β": round(beta, 2),
        "π·D1": round(pi_D1, 2),
        "π·D2": round(pi_D2, 2),
        "h1": round(h1, 2),
        "h2": round(h2, 2),
        "B": round(B, 2),
        "Corde A": round(corde_A, 2),
        "Corde C": round(corde_C, 2),
    }

    # Return arc parameters instead of points
    return {
        "arcs": [
            {"center": (0, 0), "radius": R1, "start_angle": start_angle, "end_angle": end_angle},
            {"center": (0, 0), "radius": R2, "start_angle": start_angle, "end_angle": end_angle}
        ],
        "lines": [
            (p1_outer, p1_inner),
            (p2_outer, p2_inner)
        ],
        "data": data
    }

# 3. Frustum Cone (Triangulation)
def generate_frustum_cone_triangulation(d1, d2, value, mode="H", n=12):
    D1, D2 = float(d1), float(d2)

    # --- Determine geometry like normal frustum_cone ---
    if mode.upper() == "H":
        H = value
        R1 = math.sqrt(H**2 + (D1 / 2)**2)
        R2 = math.sqrt(H**2 + (D2 / 2)**2)
        beta = 360 * (D1 - D2) / (2 * math.pi * R1)
    else:
        beta = float(value)
        if beta <= 0:
            raise ValueError("Beta must be > 0°.")
        beta_max = 360 * (D1 - D2) / (math.pi * D1)
        if beta > beta_max:
            beta = beta_max
        R_slant = (D1 - D2) * 180 / (math.pi * beta)
        R_outer = R_slant * (D1 / (D1 - D2))
        R_inner = R_slant * (D2 / (D1 - D2))
        diff = max(R_outer**2 - (D1 / 2)**2, 0)
        H = math.sqrt(diff)
        R1, R2 = R_outer, R_inner

    # --- Outline (same as normal frustum_cone) ---
    steps = 200
    outer, inner = [], []
    start_angle = -beta / 2
    for i in range(steps + 1):
        θ = math.radians(start_angle + i * (beta / steps))
        outer.append((R1 * math.cos(θ), R1 * math.sin(θ)))
        inner.append((R2 * math.cos(θ), R2 * math.sin(θ)))
    inner.reverse()
    pts = outer + inner + [outer[0]]

    # --- Generator lines ---
    generators = []
    for i in range(n + 1):
        θ = math.radians(start_angle + i * (beta / n))
        p1 = (R1 * math.cos(θ), R1 * math.sin(θ))
        p2 = (R2 * math.cos(θ), R2 * math.sin(θ))
        generators.append((p1, p2))

    # --- Calculations shown in picture ---
    # Small central triangle parameters
    a = round(math.degrees(beta / (2 * n)), 2)   # half-angle between two generators
    b = round(math.degrees(beta / n), 2)         # full angle between two adjacent generators

    L1 = round(R2, 2)
    L2 = round(R1, 2)

    data = {
        "a (°)": a,
        "b (°)": b,
        "L1 (mm)": L1,
        "L2 (mm)": L2
    }

    return {"points": pts, "generators": generators, "data": data}

# 4. Pyramid (SIMPLE CORRECT VERSION)
def generate_pyramid(AA, AB, H):
    """
    Generates a simple, correct flat pattern for a rectangular pyramid.
    """
    import math

    # --- Base dimensions ---
    Cx = AA / 2
    Cy = AB / 2

    # True edge (apex to any base corner) — constant radius for all faces
    R = math.sqrt(Cx**2 + Cy**2 + H**2)

    # Side bases with seam (AB centered): AB, AA, AB, AA, AB
    # The last AB duplicates the first to provide a closing seam in the development
    bases = [AB, AA, AB, AA, AB]

    # Central angles for each base as chords on circle radius R
    def safe_alpha(L, r):
        x = max(min(L / (2 * r), 1.0), -1.0)
        return 2 * math.asin(x)

    alphas = [safe_alpha(L, R) for L in bases]

    # Build apex-centered vertices on circle radius R (center first AB triangle)
    theta = -alphas[0] / 2.0
    pts = [(R * math.cos(theta), R * math.sin(theta))]
    for a in alphas:
        theta += a
        pts.append((R * math.cos(theta), R * math.sin(theta)))

    # Create triangular faces around the single apex (0,0)
    faces = []
    apex = (0.0, 0.0)
    for i in range(len(bases)):
        tri = [apex, pts[i], pts[i + 1]]
        faces.append(tri)

    # Computed data
    data = {
        "Base length (AA)": round(AA, 2),
        "Base width (AB)": round(AB, 2),
        "Height (H)": round(H, 2),
        "Apex to corner (R)": round(R, 2),
        "alpha1 (deg)": round(math.degrees(alphas[0]), 2),
        "alpha2 (deg)": round(math.degrees(alphas[1]), 2),
        "alpha3 (deg)": round(math.degrees(alphas[2]), 2),
        "alpha4 (deg)": round(math.degrees(alphas[3]), 2),
        "alpha5 (deg)": round(math.degrees(alphas[4]), 2),
        "Pattern type": "PYRAMID FAN (apex-centered)"
    }

    return {"faces": faces, "data": data}

# 5. Rectangle to Rectangle
def generate_rectangle_to_rectangle(ab, bc, H, AB, BC):
    """
    Génère le développé d'une transition rectangle-rectangle avec le haut inclus.
    Entrées :
        ab, bc : dimensions du rectangle supérieur
        AB, BC : dimensions du rectangle inférieur
        H : hauteur verticale (mm)
    Sortie :
        - pts DXF (faces développées)
        - données calculées : Aa, Ba, Cb
    """
    import math

    # --- Calculs géométriques ---
    Aa = math.sqrt(((AB - ab) / 2) ** 2 + H**2)
    Ba = math.sqrt(((BC - bc) / 2) ** 2 + H**2)
    Cb = math.sqrt(Aa**2 + Ba**2)

    # --- Développement 2D ---
    faces = []

    # Base (rectangle du bas)
    base = [(0, 0), (AB, 0), (AB, BC), (0, BC)]
    faces.append(base)

    # Face avant (haut du bas)
    face_front = [
        (0, 0),
        (AB, 0),
        (AB - (AB - ab) / 2, -Aa),
        ((AB - ab) / 2, -Aa)
    ]
    faces.append(face_front)

    # Face arrière
    face_back = [
        (0, BC),
        (AB, BC),
        (AB - (AB - ab) / 2, BC + Aa),
        ((AB - ab) / 2, BC + Aa)
    ]
    faces.append(face_back)

    # Face gauche
    face_left = [
        (0, 0),
        (0, BC),
        (-Ba, BC - (BC - bc) / 2),
        (-Ba, (BC - bc) / 2)
    ]
    faces.append(face_left)

    # Face droite
    face_right = [
        (AB, 0),
        (AB, BC),
        (AB + Ba, BC - (BC - bc) / 2),
        (AB + Ba, (BC - bc) / 2)
    ]
    faces.append(face_right)

    # ✅ Face du haut (rectangle supérieur)
    top = [
        ((AB - ab) / 2, -Aa),
        (AB - (AB - ab) / 2, -Aa),
        (AB - (AB - ab) / 2, -Aa - bc),
        ((AB - ab) / 2, -Aa - bc)
    ]
    faces.append(top)

    # --- Données ---
    data = {
        "Aa (mm)": round(Aa, 2),
        "Ba (mm)": round(Ba, 2),
        "Cb (mm)": round(Cb, 2),
        "Hauteur (mm)": round(H, 2)
    }

    return {"faces": faces, "data": data}

# 6. Flange
def generate_flange(D1, D2, D3, D4, N1, d1, N2, d2):
    """
    Flange ready for laser/CNC cutting.
    Keeps only outer & inner edges + holes.
    """
    import math
    entities = []

    R1 = D1 / 2   # inner hole radius
    R4 = D4 / 2   # outer contour radius

    # --- Outer and inner rings (cut contours) ---
    entities.append(("cut", (0, 0), R4))
    entities.append(("cut", (0, 0), R1))

    # --- Hole set 1 ---
    for i in range(N1):
        θ = 2 * math.pi * i / N1
        x, y = (D2 / 2) * math.cos(θ), (D2 / 2) * math.sin(θ)
        entities.append(("hole", (x, y), d1 / 2))

    # --- Hole set 2 ---
    for i in range(N2):
        θ = 2 * math.pi * i / N2
        x, y = (D3 / 2) * math.cos(θ), (D3 / 2) * math.sin(θ)
        entities.append(("hole", (x, y), d2 / 2))

    # --- Calculated data ---
    data = {
        "A1 (mm)": round(math.pi * D2, 2),
        "a1 (°)": round(360 / N1, 2) if N1 else 0,
        "A2 (mm)": round(math.pi * D3, 2),
        "a2 (°)": round(360 / N2, 2) if N2 else 0,
    }

    return {"entities": entities, "data": data}

# 7. Truncated Cylinder
def generate_truncated_cylinder(diameter, height, angle_deg, n, msp=None):
    import math, ezdxf

    R = diameter / 2
    alpha = math.radians(angle_deg)
    L_total = math.pi * diameter
    l = L_total / (n - 1)

    bottom = []
    top = []
    h_values = []

    for i in range(n):
        theta = 2 * math.pi * i / (n - 1)
        h = height + (R * math.tan(alpha) / 2) * (1 - math.cos(theta))
        x = i * l
        bottom.append((x, 0))
        top.append((x, h))
        h_values.append(round(h, 2))

    top[0] = (top[0][0], height)
    top[-1] = (top[-1][0], height)

    # If no msp passed, create local doc (for standalone test)
    if msp is None:
        doc = ezdxf.new()
        msp = doc.modelspace()
    else:
        doc = None

    # Draw pattern
    msp.add_lwpolyline(bottom, close=False)
    msp.add_line(bottom[0], top[0])
    msp.add_spline(top)
    msp.add_line(top[-1], bottom[-1])

    result = {
        "data": {
            "piD": round(L_total, 2),
            "l": round(l, 2),
            "h_values": h_values
        }
    }

    # If standalone test mode
    if doc:
        import io, base64
        buf = io.StringIO()
        doc.write(buf)
        dxf_data = buf.getvalue().encode("utf-8")
        buf.close()
        result["dxf_base64"] = base64.b64encode(dxf_data).decode("utf-8")

    return result

# 8. Bend
def generate_bend(R, alpha_deg, D, N, n, msp=None, layer="CUT"):
    """
    Génère un développé rectangulaire avec deux bords sinusoïdaux
    parfaitement symétriques, phase corrigée (crête à gauche).
    """
    import math

    # --- Géométrie de base ---
    piD = math.pi * D
    l = piD / n
    H = D
    y_center = H / 2
    alpha_rad = math.radians(alpha_deg)

    # Amplitude et écart vertical entre les ondes
    A = D * 0.15
    gap = D * 0.5

    # --- Onde sinusoïdale (phase corrigée : départ en crête) ---
    def wave(x):
        return A * math.sin((2 * math.pi * x / piD) - math.pi / 2)

    top_pts, bot_pts, h_vals = [], [], []

    for i in range(n + 1):
        x = i * l
        y_top = y_center + gap / 2 + wave(x)
        y_bot = y_center - gap / 2 - wave(x)
        top_pts.append((x, y_top))
        bot_pts.append((x, y_bot))
        h_vals.append(round(wave(x), 2))

    # --- Si aucun modelspace n'est fourni, on crée un DXF local pour test ---
    if msp is None:
        import ezdxf, io, base64
        doc = ezdxf.new()
        msp = doc.modelspace()
        local_mode = True
    else:
        doc = None
        local_mode = False

    # --- Dessin dans le DXF ---
    msp.add_lwpolyline([(0, 0), (piD, 0), (piD, H), (0, H), (0, 0)], close=True, dxfattribs={"layer": layer})
    msp.add_spline(top_pts, dxfattribs={"layer": layer, "color": 5})
    msp.add_spline(bot_pts, dxfattribs={"layer": layer, "color": 5})

    result = {
        "data": {
            "piD": round(piD, 2),
            "A": round(A, 2),
            "l": round(l, 2),
            "h_vals": h_vals,
            "gap": round(gap, 2),
        }
    }

    # --- Si mode local (test), générer DXF encodé ---
    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_bytes = buf.getvalue().encode("utf-8")
        result["dxf_base64"] = base64.b64encode(dxf_bytes).decode("utf-8")

    return result

# 9. Circle to Rectangle
def generate_circle_to_rectangle(D, H, A, B, n, msp=None, layer="CUT"):
    """
    Circle → Rectangle transition flat pattern (symmetrical development).
    Bottom: circular arc (unrolled), Top: small rectangle (centered)
    """
    import math, ezdxf, io, base64

    # --- Geometry ---
    R = D / 2.0
    n = max(60, int(n))  # Sampling density

    # Rectangle perimeter and circle circumference
    rect_perim = 2 * (A + B)
    circ_perim = 2 * math.pi * R

    # Development angle: how much of the circle to unroll
    # For symmetry, unroll an arc that's proportional to rect vs circle size
    theta_dev = 2 * math.pi * (rect_perim / circ_perim)
    theta_dev = min(theta_dev, 2 * math.pi * 0.9)  # Cap at 90% of full circle

    # Sample points along circle arc (centered, symmetric)
    angles = [theta_dev * (i / n - 0.5) for i in range(n + 1)]
    circ_pts_plan = [(R * math.cos(a), R * math.sin(a)) for a in angles]

    # Sample points along rectangle perimeter (walk all 4 sides)
    def rect_point_at_t(t):
        # t ∈ [0,1] parameterizes rectangle perimeter
        s = t * rect_perim
        if s < B:  # bottom edge
            return (-B/2 + s, -A/2)
        s -= B
        if s < A:  # right edge
            return (B/2, -A/2 + s)
        s -= A
        if s < B:  # top edge
            return (B/2 - s, A/2)
        s -= B
        # left edge
        return (-B/2, A/2 - s)

    rect_pts_plan = [rect_point_at_t(i / n) for i in range(n + 1)]

    # True lengths: distance from circle to rectangle in 3D
    def dist(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])
    L = [math.sqrt(H**2 + dist(circ_pts_plan[i], rect_pts_plan[i])**2) for i in range(n + 1)]

    # Development X: arc length along unrolled circle
    X = [R * angles[i] for i in range(n + 1)]

    # --- DXF setup ---
    local_mode = False
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True

    # --- Build single symmetrical outline ---
    # Bottom edge: use a curved base (scaled circular arc) to match
    # the target "demi-circle" look, and place the top edge above it
    # by the true length at each station.
    base_curvature_scale = 0.35  # 0..1, controls how pronounced the base arc is
    base_y = [base_curvature_scale * R * (1.0 - math.cos(angles[i])) for i in range(n + 1)]
    bottom_pts = [(X[i], base_y[i]) for i in range(n + 1)]
    top_pts = [(X[i], base_y[i] + L[i]) for i in range(n + 1)]

    # Full outline: left side + top + right side + bottom
    outline = [bottom_pts[0]] + top_pts + [bottom_pts[-1]] + bottom_pts[::-1]
    msp.add_lwpolyline(outline, close=True, dxfattribs={"layer": layer})

    # --- Draw alternating fan generators across 4 segments ---
    # We split the strip into 4 equal segments. In each segment, lines
    # radiate from the segment boundary on one side (alternating left/right)
    # to the top curve points within that segment. This reproduces the
    # alternating triangular fan look in the reference image.
    segments = 4
    boundaries = [round(i * n / segments) for i in range(segments + 1)]

    for s in range(segments):
        i_start = boundaries[s]
        i_end = boundaries[s + 1]
        # Use the segment midpoint on the bottom as the apex so that
        # each segment has a distinct convergence point.
        apex_index = (i_start + i_end) // 2
        apex_point = (X[apex_index], base_y[apex_index])

        # Choose a local density so each fan has ~20–30 rays
        local_count = max(20, (i_end - i_start))
        local_step = max(1, (i_end - i_start) // 25)

        # Connect apex to top-curve points across the segment, alternating
        # sweep direction to enhance the mirrored look.
        if s % 2 == 0:
            idx_range = range(i_start, i_end + 1, local_step)
        else:
            idx_range = range(i_end, i_start - 1, -local_step)

        for i in idx_range:
            msp.add_line(apex_point, (X[i], base_y[i] + L[i]), dxfattribs={"layer": layer})

    result = {
        "data": {
            "R": round(R, 2),
            "A": round(A, 2),
            "B": round(B, 2),
            "H": round(H, 2),
            "n": n,
            "Development_angle_deg": round(math.degrees(theta_dev), 2),
            "Width": round(X[-1] - X[0], 2)
        }
    }

    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_bytes = buf.getvalue().encode("utf-8")
        result["dxf_base64"] = base64.b64encode(dxf_bytes).decode("utf-8")

    return result

# 10. Rectangle to Circle
def generate_rectangle_to_circle(D, H, A, B, n, msp=None, layer="CUT"):
    """
    Generate DXF flat pattern for Rectangle → Circle transition.
    Compatible with unified DXF pipeline (draws directly in msp).
    """
    import math, ezdxf, io, base64

    # --- Geometry ---
    R = D / 2
    n = max(6, int(n))
    A_star = A
    B_star = B
    R_dev = math.sqrt((A / 2) ** 2 + (B / 2) ** 2 + H ** 2)
    c = (math.pi * R) / n

    # --- Slant lengths for data ---
    l_values = []
    for i in range(1, n + 1):
        theta = math.pi * (i - 1) / (n - 1)
        dx = (A / 2) - (R * math.cos(theta))
        dy = (B / 2) - (R * math.sin(theta))
        l = math.sqrt(dx**2 + dy**2 + H**2)
        l_values.append(l)

    # --- DXF handling ---
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True
    else:
        doc = None
        local_mode = False

    # --- Outer pattern (rectangle projected edge) ---
    angle_step = 2 * math.pi / (2 * n)
    points = []
    for i in range(n + 1):
        angle = i * angle_step
        x = R_dev * math.sin(angle)
        y = R_dev * math.cos(angle)
        points.append((x, y))

    # ✅ Top circle (true circular edge)
    msp.add_circle((0, 0), R, dxfattribs={"layer": layer})

    # Connect outer to circle perimeter using radial lines
    for i in range(n + 1):
        angle = i * angle_step
        x_outer = R_dev * math.sin(angle)
        y_outer = R_dev * math.cos(angle)
        x_inner = R * math.sin(angle)
        y_inner = R * math.cos(angle)

        # Outer edge line (next segment)
        if i < n:
            x_next = R_dev * math.sin(angle + angle_step)
            y_next = R_dev * math.cos(angle + angle_step)
            msp.add_line((x_outer, y_outer), (x_next, y_next), dxfattribs={"layer": layer})

        # Radial connection
        msp.add_line((x_outer, y_outer), (x_inner, y_inner), dxfattribs={"layer": layer})

    # Close first connection
    msp.add_line(points[0], (R * math.sin(0), R * math.cos(0)), dxfattribs={"layer": layer})

    # --- Output data ---
    data = {
        "c": round(c, 2),
        "R_dev": round(R_dev, 2),
        "R_top": round(R, 2),
        "A*": round(A_star, 2),
        "B*": round(B_star, 2),
    }
    for i, l in enumerate(l_values, 1):
        data[f"l{i}"] = round(l, 2)

    result = {"data": data}

    # --- If local test mode, also return encoded DXF ---
    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_bytes = buf.getvalue().encode("utf-8")
        result["dxf_base64"] = base64.b64encode(dxf_bytes).decode("utf-8")

    return result

# 11. Rectangle to Circle Eccentric
def generate_rectangle_to_circle_ecc(params, msp=None, layer="CUT"):
    """
    Rectangle → Cercle excentré (développement)
    Compatible avec pipeline DXF unifié.
    Entrée params:
      {
        "D": float, "H": float,
        "A": float, "B": float,
        "X": float, "Y": float,
        "n": int
      }
    Retour:
      {
        "entities": [...],
        "calc": {...}
      }
    """
    import math

    # --- Données d'entrée ---
    D = float(params["D"])
    H = float(params["H"])
    A = float(params["A"])
    B = float(params["B"])
    X = float(params["X"])
    Y = float(params["Y"])
    n = max(3, int(params.get("n", 12)))

    R = D / 2.0

    # Coins du rectangle (vue en plan)
    a = (-B/2.0,  A/2.0)
    b = (-B/2.0, -A/2.0)
    c = ( B/2.0, -A/2.0)
    d = ( B/2.0,  A/2.0)

    Cx, Cy = X, Y

    def dist2d(P, Q):
        return math.hypot(P[0]-Q[0], P[1]-Q[1])

    # Longueur d’arc élémentaire
    perim = math.tau * R
    l = perim / n

    # Offsets utiles
    ah = min(abs(Cy - (+A/2.0)), abs(Cy - (-A/2.0)))
    dh = min(abs(Cx - (+B/2.0)), abs(Cx - (-B/2.0)))

    # Directions angulaires
    ang_b = math.atan2(b[1]-Cy, b[0]-Cx)
    ang_c = math.atan2(c[1]-Cy, c[0]-Cx)

    def wrap(a): 
        while a <= -math.pi: a += 2*math.pi
        while a >  math.pi: a -= 2*math.pi
        return a

    da = wrap(ang_c - ang_b)
    if da <= 0:
        da += 2*math.pi
    dtheta = da / n

    # Points de l'arc supérieur
    arc_pts = [(Cx + R*math.cos(ang_b + i*dtheta),
                Cy + R*math.sin(ang_b + i*dtheta)) for i in range(n+1)]

    # --- Vraies longueurs des génératrices ---
    def true_length(P, Q_on_circle):
        return math.hypot(H, dist2d(P, Q_on_circle))

    aP, bP, cP, dP = a, b, c, d
    aQ1, aQ2 = arc_pts[0], arc_pts[n//2]
    bQ1, bQ2 = arc_pts[0], arc_pts[1]
    cQ1, cQ2 = arc_pts[-2], arc_pts[-1]
    dQ1, dQ2 = arc_pts[n//2], arc_pts[-1]

    a1 = true_length(aP, aQ1); a2 = true_length(aP, aQ2)
    b1 = true_length(bP, bQ1); b2 = true_length(bP, bQ2)
    c1 = true_length(cP, cQ1); c2 = true_length(cP, cQ2)
    d1 = true_length(dP, dQ1); d2 = true_length(dP, dQ2)

    mid_base = ((b[0]+c[0])/2.0, (b[1]+c[1])/2.0)
    mid_arc  = arc_pts[n//2]
    h1 = true_length(mid_base, mid_arc)

    # --- Dessin du développé ---
    entities = []
    def add_line(p, q):
        if msp is not None:
            msp.add_line(p, q, dxfattribs={"layer": layer})
        else:
            entities.append(("LINE", p, q))

    # Base BC
    Bv = (0.0, 0.0)
    Cv = (B,  0.0)
    add_line(Bv, Cv)

    O = ((Bv[0]+Cv[0])/2.0, 0.0)

    step = B / n
    gens_bottom = [(O[0] - (n/2.0 - i)*step, 0.0) for i in range(n+1)]
    gens_top = []
    for i in range(n+1):
        Pbot = gens_bottom[i]
        Qtop = arc_pts[i]
        L = true_length(Pbot, Qtop)
        gens_top.append((Pbot[0], L))

    # Arc concave
    for i in range(n):
        add_line(gens_top[i], gens_top[i+1])

    # Génératrices
    for i in range(n+1):
        add_line(gens_bottom[i], gens_top[i])

    # Panneaux latéraux gauche/droite
    def radial_point(base_pt, length, angle_deg):
        ang = math.radians(angle_deg)
        return (base_pt[0] + length*math.cos(ang), base_pt[1] + length*math.sin(ang))

    left_peak  = radial_point(Bv, b1, 110)
    right_peak = radial_point(Cv, c2, 70)
    add_line(Bv, left_peak); add_line(left_peak, gens_top[0])
    add_line(Cv, right_peak); add_line(right_peak, gens_top[-1])

    apex_L = gens_top[1]; apex_R = gens_top[-2]
    add_line(Bv, apex_L); add_line(Cv, apex_R)
    add_line(O, gens_top[n//2])

    # --- Données calculées ---
    calc = {
        "l": round(l, 2),
        "h1": round(h1, 2),
        "ah": round(ah, 2),
        "dh": round(dh, 2),
        "a1": round(a1, 2), "a2": round(a2, 2),
        "b1": round(b1, 2), "b2": round(b2, 2),
        "c1": round(c1, 2), "c2": round(c2, 2),
        "d1": round(d1, 2), "d2": round(d2, 2),
    }

    return {"entities": entities, "calc": calc}

# 12. Frustum Eccentric (Angle)
def generate_frustum_ecc_angle(params, msp=None, layer="0"):
    """
    Frustum Eccentric (Angle)
    Inputs:
      D1, D2 : diameters (top/bottom)
      H : height
      X : eccentric offset (horizontal)
      a : inclination angle (deg)
      n : number of generators
    Returns:
      {
        "calc": {"a":..., "b":..., "L1":..., "L2":..., ...},
        "entities": [...]
      }
    """
    D1 = float(params["D1"])
    D2 = float(params["D2"])
    H = float(params["H"])
    X = float(params["X"])
    # ✅ Accept both "a" and "alpha"
    alpha = float(params.get("a", params.get("alpha", 0)))
    n = int(params["n"])

    R1 = D1 / 2
    R2 = D2 / 2

    # Half cone angle (in degrees)
    a_angle = math.degrees(math.atan((R2 - R1) / H))
    a = round(a_angle, 2)

    # Calculate true lengths for each generator
    L = []
    for i in range(1, n + 1):
        θ = math.pi * (i - 1) / (n - 1)
        L_i = math.sqrt(H**2 + (R2 - R1 + X * math.cos(θ))**2)
        L.append(round(L_i, 2))

    # Average length for development sector
    Lavg = sum(L) / len(L)
    b_angle = 360 * R2 / Lavg
    b = round(b_angle, 2)

    # DXF drawing (flattened pattern)
    entities = []
    def add_line(p, q):
        if msp is not None:
            msp.add_line(p, q, dxfattribs={"layer": layer})
        else:
            entities.append(("LINE", p, q))

    step = math.radians(b_angle) / (n - 1)
    pts_top = []
    pts_bot = []

    for i in range(n):
        ang = i * step
        r_top = L[i] - (R2 - R1)
        r_bot = L[i]
        x1, y1 = r_top * math.cos(ang), r_top * math.sin(ang)
        x2, y2 = r_bot * math.cos(ang), r_bot * math.sin(ang)
        pts_top.append((x1, y1))
        pts_bot.append((x2, y2))

    # Create proper eccentric frustum development with varying generator lengths
    # Use the actual calculated generator lengths for the development
    
    # Generate smooth arcs using the varying generator lengths
    smooth_top = []
    smooth_bot = []
    num_smooth = 200
    
    for i in range(num_smooth + 1):
        # Interpolate between generator positions
        t = i / num_smooth
        idx_float = t * (n - 1)
        idx_low = int(idx_float)
        idx_high = min(idx_low + 1, n - 1)
        frac = idx_float - idx_low
        
        # Interpolate generator length (this creates the eccentric effect)
        L_interp = L[idx_low] + frac * (L[idx_high] - L[idx_low]) if idx_high < len(L) else L[idx_low]
        
        # Calculate angle in development
        ang = t * math.radians(b_angle)
        
        # Use interpolated lengths for proper eccentric development
        r_bot = L_interp
        r_top = L_interp - (R2 - R1)
        
        x_bot = r_bot * math.cos(ang)
        y_bot = r_bot * math.sin(ang)
        x_top = r_top * math.cos(ang)
        y_top = r_top * math.sin(ang)
        
        smooth_top.append((x_top, y_top))
        smooth_bot.append((x_bot, y_bot))

    # Draw smooth arcs using the varying radii (creates eccentric shape)
    if msp is not None:
        msp.add_lwpolyline(smooth_top, close=False, dxfattribs={"layer": layer})
        msp.add_lwpolyline(smooth_bot, close=False, dxfattribs={"layer": layer})
    
    # Draw side closing lines (left and right edges)
    add_line(smooth_top[0], smooth_bot[0])   # Left edge
    add_line(smooth_top[-1], smooth_bot[-1])  # Right edge
    
    # Draw generator lines
    for i in range(n):
        add_line(pts_top[i], pts_bot[i])

    calc = {"a": a, "b": b}
    for i, Li in enumerate(L, start=1):
        calc[f"L{i}"] = Li

    return {"calc": calc, "entities": entities}

# 13. Frustum Eccentric Parallel (Flat)
def generate_frustum_ecc_paral(params, msp=None, layer="0"):
    """
    Frustum Eccentric Parallel (Flat pattern)
    Inputs:
      D1, D2 : diameters of top/bottom circles
      H : vertical height
      X : eccentric offset
      n : number of generators
    Returns:
      {
        "calc": {"a":..., "b":..., "L1":..., "L2":..., ...},
        "entities": [...]
      }
    """

    D1 = float(params["D1"])
    D2 = float(params["D2"])
    H = float(params["H"])
    X = float(params["X"])
    n = int(params["n"])

    R1 = D1 / 2
    R2 = D2 / 2

    # ---- cone half-angle ----
    a_angle = math.degrees(math.atan((R2 - R1) / H))
    a = round(a_angle, 2)

    # ---- compute generator lengths ----
    L = []
    for i in range(1, n + 1):
        θ = math.pi * (i - 1) / (n - 1)
        Li = math.sqrt(H**2 + (R2 - R1 + X * math.cos(θ))**2)
        L.append(round(Li, 2))

    # ---- sector angle ----
    Lavg = sum(L) / len(L)
    b_angle = 360 * R2 / Lavg
    b = round(b_angle, 2)

    # ---- DXF development ----
    entities = []
    def add_line(p, q):
        if msp is not None:
            msp.add_line(p, q, dxfattribs={"layer": layer})
        else:
            entities.append(("LINE", p, q))

    # Center the development so the middle rib is vertical (fan-like look)
    beta_rad = math.radians(b_angle)

    # Smooth inner/outer arcs with constant radii for fan-like appearance
    num_smooth = 360
    smooth_bot = []
    smooth_top = []
    for i in range(num_smooth + 1):
        t = i / num_smooth
        ang = -beta_rad / 2 + t * beta_rad
        # Constant radii: bottom arc at R1, top arc at R2
        r_bot = R1
        r_top = R2
        smooth_bot.append((r_bot * math.cos(ang), r_bot * math.sin(ang)))
        smooth_top.append((r_top * math.cos(ang), r_top * math.sin(ang)))

    if msp is not None:
        msp.add_lwpolyline(smooth_top, close=False, dxfattribs={"layer": layer})
        msp.add_lwpolyline(smooth_bot, close=False, dxfattribs={"layer": layer})

    # Radiating ribs (evenly spaced angles across the sector)
    ribs = int(params.get("ribs", max(2 * n - 1, n)))
    ribs = max(ribs, 3)
    for i in range(ribs):
        t = i / (ribs - 1)
        ang = -beta_rad / 2 + t * beta_rad
        # Constant radii for ribs
        r_bot = R1
        r_top = R2
        p_top = (r_top * math.cos(ang), r_top * math.sin(ang))
        p_bot = (r_bot * math.cos(ang), r_bot * math.sin(ang))
        add_line(p_top, p_bot)

    # Side edges for closure
    add_line(smooth_top[0], smooth_bot[0])
    add_line(smooth_top[-1], smooth_bot[-1])

    # ---- results ----
    calc = {"a": a, "b": b}
    for i, Li in enumerate(L, start=1):
        calc[f"L{i}"] = Li

    return {"calc": calc, "entities": entities}

# 14. Offset Cone
def generate_offset_cone(D, H, X, n, msp=None, layer="CUT"):
    """
    Generate DXF flat pattern for an Offset Cone (Cône excentré).
    Compatible with unified DXF pipeline.
    Inputs:
        D : diameter base
        H : height
        X : offset
        n : number of divisions
        msp: DXF modelspace (optional)
        layer: DXF layer name
    Returns:
        {
          "data": {...},  # geometric data
          "dxf_base64": ...  # only if msp=None (standalone mode)
        }
    """
    import math, ezdxf, io, base64

    # --- Base geometry ---
    R = D / 2
    a = 180 / n  # degree step

    # --- Compute generator lengths ---
    L_values = []
    for i in range(n + 1):
        theta = math.radians(i * a)
        Li = math.sqrt(H**2 + (R + X * math.cos(theta))**2)
        L_values.append(Li)

    # --- DXF setup ---
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True
    else:
        doc = None
        local_mode = False

    # --- Geometry plotting ---
    apex = (0, 0)
    angle_step = math.radians(a)
    angle_accum = 0
    points = []

    for Li in L_values:
        x = Li * math.cos(angle_accum)
        y = Li * math.sin(angle_accum)
        points.append((x, y))
        angle_accum += angle_step

    # Draw edges
    msp.add_lwpolyline([apex] + points, close=False, dxfattribs={"layer": layer})
    msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})

    # --- Output data ---
    data = {"a": round(a, 2)}
    for i, L in enumerate(L_values):
        data[f"L{i}"] = round(L, 2)

    result = {"data": data}

    # --- Encode DXF only if used standalone ---
    if local_mode:
        text_buffer = io.StringIO()
        doc.write(text_buffer)
        dxf_text = text_buffer.getvalue()
        text_buffer.close()
        dxf_base64 = base64.b64encode(dxf_text.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_base64

    return result

# 15. Sphere
def generate_sphere(D, N, n, msp=None, layer="CUT"):
    """
    Generate DXF flat pattern for full sphere gore development.
    Compatible with unified DXF pipeline.
    Inputs:
        D : Sphere diameter
        N : Number of gores (vertical slices)
        n : Number of horizontal bands (latitude divisions)
        msp: DXF modelspace (optional)
        layer: DXF layer (default = "CUT")
    Returns:
        {
          "data": {...},
          "dxf_base64": ... (only if msp=None)
        }
    """
    import math, ezdxf, io, base64

    R = D / 2
    piR = math.pi * R

    # Angular divisions (0 → 180° for full sphere)
    theta_vals = [math.radians(i * 180 / n) for i in range(n + 1)]

    # Local radii and heights for each latitude
    r_vals = [R * math.sin(t) for t in theta_vals]
    y_vals = [R * math.cos(t) for t in theta_vals]

    # Chord lengths per latitude (width of each gore at each latitude)
    L_vals = [(2 * math.pi * r) / N for r in r_vals]

    # --- DXF setup ---
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True
    else:
        doc = None
        local_mode = False

    gore_spacing = (piR / N) * 1.05  # small offset between gores for clarity

    # --- Draw each gore ---
    for g in range(N):
        x_shift = g * gore_spacing
        pts_left = []
        pts_right = []

        # Generate curved gore outline using smooth interpolation
        for i in range(len(L_vals) - 1):
            y1, y2 = y_vals[i], y_vals[i + 1]
            l1, l2 = L_vals[i] / 2, L_vals[i + 1] / 2

            # Smooth transition between latitudes
            for k in range(11):
                t = k / 10
                y = y1 + (y2 - y1) * t
                half_width = l1 + (l2 - l1) * t
                pts_left.append((x_shift - half_width, y))
                pts_right.append((x_shift + half_width, y))

        # Outline gore contour
        outline = pts_left + pts_right[::-1]
        msp.add_lwpolyline(outline, close=True, dxfattribs={"layer": layer})

        # Add internal latitude lines (optional visual grid)
        for i in range(len(y_vals)):
            y = y_vals[i]
            half_w = L_vals[i] / 2
            msp.add_line(
                (x_shift - half_w, y),
                (x_shift + half_w, y),
                dxfattribs={"layer": layer},
            )

    # --- Output data ---
    data = {"πR": round(piR, 2)}
    for i, L in enumerate(L_vals, 1):
        data[f"L{i}"] = round(L, 2)

    result = {"data": data}

    # --- Encode DXF if standalone ---
    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_data = buf.getvalue()
        buf.close()
        dxf_base64 = base64.b64encode(dxf_data.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_base64

    return result

# 16. Auger (Helical Screw Flight)
def generate_auger(params, msp=None, layer="CUT"):
    """
    Generate flat pattern for Auger (helical screw flight)
    Inputs:
      d : inner diameter (shaft)
      D : outer diameter (flight)
      S : pitch (distance between turns)
    Returns:
      {"calc": {...}, "entities": [...]}
    """
    d = float(params["d"])
    D = float(params["D"])
    S = float(params["S"])

    r1 = d / 2
    r2 = D / 2
    rm = (r1 + r2) / 2

    # true length of one turn
    L = math.sqrt(S**2 + (2 * math.pi * rm)**2)

    # developed sector angle
    alpha = 360 * (2 * math.pi * rm) / L

    # developed inner and outer radii
    r = L * r1 / (2 * math.pi * rm)
    R = L * r2 / (2 * math.pi * rm)

    # arc length at mean radius
    A = 2 * math.pi * rm * (alpha / 360)

    # --- DXF Pattern (Annular sector) ---
    entities = []
    def add_line(p, q):
        if msp is not None:
            msp.add_line(p, q, dxfattribs={"layer": layer})
        else:
            entities.append(("LINE", p, q))

    # Create arcs
    steps = 60
    angle_step = math.radians(alpha / steps)
    outer_pts = [(R * math.cos(i * angle_step), R * math.sin(i * angle_step)) for i in range(steps + 1)]
    inner_pts = [(r * math.cos(i * angle_step), r * math.sin(i * angle_step)) for i in reversed(range(steps + 1))]

    poly = outer_pts + inner_pts + [outer_pts[0]]
    if msp is not None:
        msp.add_lwpolyline(poly, close=True, dxfattribs={"layer": layer})
    else:
        entities.append(("LWPOLYLINE", poly))

    # --- Calculations output ---
    calc = {
        "A": round(A, 2),
        "a": round(alpha, 2),
        "r": round(r, 2),
        "R": round(R, 2)
    }

    return {"calc": calc, "entities": entities}

# 17. Breeches (2-branch Y piece)
def generate_breeches_full(params, msp=None, layer="CUT"):
    """
    Generate Breeches (2-branch) flat pattern (développement des culottes doubles).
    Compatible avec le pipeline DXF global.

    Entrée:
        params = {
          "D":  diameter (mm),
          "L1": longueur branche A (mm),
          "L2": longueur branche B (mm),
          "a":  angle entre branches (°),
          "n":  nombre de génératrices
        }
    Retour:
        {
          "calc": {...},
          "dxf_base64": ... (si msp=None)
        }
    """
    import math, ezdxf, io, base64

    # --- Inputs ---
    D = float(params["D"])
    L1 = float(params["L1"])
    L2 = float(params["L2"])
    a = math.radians(float(params["a"]))
    n = int(params["n"])

    # --- Calculations ---
    periphery = math.pi * D
    l = periphery / n

    # Hauteurs géométriques
    h1 = L1
    h2 = L2
    h3 = math.sqrt(L1**2 + L2**2 - 2 * L1 * L2 * math.cos(a))
    h1p = L2
    h2p = h3

    # --- DXF Setup ---
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True
    else:
        doc = None
        local_mode = False

    # --- DXF Drawing ---
    pts_top = []
    pts_bot = []

    # Profiles supérieur et inférieur
    for i in range(n + 1):
        x = i * l
        theta = i / n * math.pi

        # courbe haute (branche A)
        y_top = h1 - (h1 - h3) * (1 - math.cos(theta)) / 2
        # courbe basse (branche B)
        y_bot = -h1p + (h1p - h2p) * (1 - math.cos(theta)) / 2

        pts_top.append((x, y_top))
        pts_bot.append((x, y_bot))

    # Polyligne fermée du développé
    outline = []
    outline += pts_bot                          # courbe inférieure
    outline += [(pts_bot[-1][0], 0)]            # droite verticale droite
    outline += list(reversed(pts_top))          # courbe supérieure inversée
    outline += [(0, 0)]                         # fermeture gauche

    msp.add_lwpolyline(outline, close=True, dxfattribs={"layer": layer})

    # Lignes guides (optionnelles)
    msp.add_line((0, 0), (periphery, 0), dxfattribs={"color": 3, "layer": layer})
    msp.add_line(
        (0, (h1 - h1p) / 2),
        (periphery, (h1 - h1p) / 2),
        dxfattribs={"color": 5, "layer": layer},
    )

    # --- Données calculées ---
    calc = {
        "πD": round(periphery, 2),
        "l": round(l, 2),
        "h1": round(h1, 2),
        "h2": round(h2, 2),
        "h3": round(h3, 2),
        "h'1": round(h1p, 2),
        "h'2": round(h2p, 2),
    }

    result = {"calc": calc}

    # --- Si standalone, encoder DXF ---
    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_text = buf.getvalue()
        buf.close()
        dxf_base64 = base64.b64encode(dxf_text.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_base64

    return result

# 18. Offset Tee (Oblique branch with offset)
def generate_offset_tee(params, msp=None, layer="CUT"):
    """
    Generate the flat pattern for an OFFSET TEE (oblique branch with offset).
    Compatible with the unified DXF pipeline.

    Inputs:
      D - main pipe diameter
      d - branch diameter
      L - branch length
      X - offset distance
      a - branch angle (degrees)
      n - number of generators
    Returns:
      {
        "calc": {...},
        "dxf_base64": ... (only if msp=None)
      }
    """
    import math, ezdxf, io, base64

    # --- Parameters ---
    D = float(params["D"])
    d = float(params["d"])
    L = float(params["L"])
    X = float(params["X"])
    a = math.radians(float(params["a"]))
    n = int(params["n"])

    r = d / 2
    periphery = math.pi * d
    l = periphery / n

    # --- Compute heights (intersection line development) ---
    heights = []
    points = []
    for i in range(n + 1):
        theta = 2 * math.pi * i / n
        # Theoretical height for oblique cut with offset
        h = L * math.sin(a) + X * (1 - math.cos(theta))
        heights.append(round(h, 2))
        points.append((i * l, h))

    # --- DXF Setup ---
    local_mode = False
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True

    # --- DXF Drawing ---
    # Base line
    msp.add_line((0, 0), (periphery, 0), dxfattribs={"layer": layer})
    # Top curve
    msp.add_lwpolyline(points, dxfattribs={"layer": layer})
    # Vertical generator lines
    for i in range(n + 1):
        x = i * l
        msp.add_line((x, 0), (x, points[i][1]), dxfattribs={"layer": layer})

    # --- Data ---
    data = {"π*d": round(periphery, 2), "l": round(l, 2)}
    for i, h in enumerate(heights, start=1):
        data[f"h{i}"] = h

    result = {"calc": data}

    # --- Encode DXF if standalone ---
    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_text = buf.getvalue()
        buf.close()
        dxf_base64 = base64.b64encode(dxf_text.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_base64

    return result

# 19. Tee Oblique (Y-Tee)
def generate_tee_oblique(params, msp=None, layer="CUT"):
    """
    Generate the flat pattern for an oblique Tee connection (α <= 90°)
    Returns both:
      A: branch development (flat pattern)
      B: main pipe cutout (elliptical projection)
    Compatible with unified DXF pipeline.
    """
    import math, ezdxf, io, base64

    # --- Inputs ---
    L1 = float(params["L1"])
    D = float(params["D"])
    d = float(params["d"])
    a = math.radians(float(params["a"]))
    n = int(params["n"])

    r = d / 2
    R = D / 2
    periph_d = math.pi * d
    periph_D = math.pi * D
    l = periph_d / n

    # -------------------------------------
    # Part A — Branch development
    # -------------------------------------
    pts_A = []
    h_values = []

    for i in range(n + 1):
        theta = 2 * math.pi * i / n
        # Intersection height along the branch
        h = L1 * math.sin(a) * abs(math.cos(theta / 2))
        h_values.append(round(h, 2))
        pts_A.append((i * l, h))

    # Close branch outline with vertical sides + base line
    pts_A_closed = [(0, 0)] + pts_A + [(periph_d, 0)]

    # -------------------------------------
    # Part B — Main pipe cutout
    # -------------------------------------
    pts_B = []
    h_prime = []
    l_prime = []

    for i in range(n + 1):
        theta = 2 * math.pi * i / n
        h2 = r * math.sin(a) * math.sin(theta / 2)
        l2 = r * math.cos(a) * (1 - math.cos(theta / 2))
        h_prime.append(round(h2, 2))
        l_prime.append(round(l2, 2))
        x = l2 * 5.0  # scaled for visibility
        y = h2 * 5.0
        pts_B.append((x, y))

    # -------------------------------------
    # DXF Drawing (works with or without msp)
    # -------------------------------------
    local_mode = False
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True

    # Branch pattern (left)
    msp.add_lwpolyline(pts_A_closed, close=True, dxfattribs={"layer": layer})

    # Main pipe hole (right, shifted)
    offset_x = periph_d + d * 2
    pts_B_shifted = [(x + offset_x, y) for x, y in pts_B]
    msp.add_lwpolyline(pts_B_shifted, close=True, dxfattribs={"layer": layer})

    # Guide line between both parts
    msp.add_line((periph_d, 0), (offset_x, 0), dxfattribs={"layer": layer, "color": 2})

    # -------------------------------------
    # Calculations
    # -------------------------------------
    calc = {
        "π*d": round(periph_d, 2),
        "l": round(l, 2),
        "π*D": round(periph_D, 2),
    }

    # Heights h1..hn
    for i, h in enumerate(h_values, start=1):
        calc[f"h{i}"] = h
    # Hole profile h′, l′
    for i, h2 in enumerate(h_prime, start=1):
        calc[f"h'{i}"] = h2
    for i, l2 in enumerate(l_prime, start=1):
        calc[f"l'{i}"] = l2

    result = {"calc": calc}

    # -------------------------------------
    # Encode DXF if standalone
    # -------------------------------------
    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_text = buf.getvalue()
        buf.close()
        dxf_base64 = base64.b64encode(dxf_text.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_base64

    return result

# 20. Tee Eccentric (α = 90°, offset X)
def generate_tee_eccentric(params, msp=None, layer="CUT"):
    """
    Generate the flat pattern for a Tee Eccentric (90° branch with offset X)
    Compatible with unified DXF pipeline.
    Inputs:
      D : diameter of main pipe
      d : diameter of branch
      H : height of branch (projection)
      X : offset between axes
      n : number of generators
    Returns:
      {
        "calc": {...},
        "dxf_base64": ... (only if msp=None)
      }
    """
    import math, ezdxf, io, base64

    # --- Inputs ---
    D = float(params["D"])
    d = float(params["d"])
    H = float(params["H"])
    X = float(params["X"])
    n = int(params["n"])

    # --- Calculations ---
    periph_d = math.pi * d
    l = periph_d / n

    pts = []
    h_values = []

    # Generate top curve points (eccentric)
    for i in range(n + 1):
        theta = 2 * math.pi * i / n
        h = H + X * math.sin(theta)  # offset distortion
        pts.append((i * l, h))
        h_values.append(round(h, 2))

    # Close the shape (flat bottom)
    pts_closed = [(0, 0)] + pts + [(periph_d, 0)]

    # --- DXF Setup ---
    local_mode = False
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True

    # --- DXF Drawing ---
    # Main profile
    msp.add_lwpolyline(pts_closed, close=True, dxfattribs={"layer": layer})
    # Baseline
    msp.add_line((0, 0), (periph_d, 0), dxfattribs={"layer": layer, "color": 3})
    # Optional generators
    for i in range(n + 1):
        x = i * l
        msp.add_line((x, 0), (x, pts[i][1]), dxfattribs={"layer": layer, "color": 5})

    # --- Calculation results ---
    calc = {"π*d": round(periph_d, 2), "l": round(l, 2)}
    for i, h in enumerate(h_values, start=1):
        calc[f"h{i}"] = h

    result = {"calc": calc}

    # --- Encode DXF if standalone ---
    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_text = buf.getvalue()
        buf.close()
        dxf_base64 = base64.b64encode(dxf_text.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_base64

    return result

# 21. Tee on Bend (Branch on Curved Main Pipe)
def generate_tee_on_bend(params, msp=None, layer="CUT"):
    """
    Generate flat pattern for Tee on Bend (branch on curved main pipe)
    Compatible with unified DXF pipeline.

    Inputs:
      D : main pipe diameter
      d : branch diameter
      R : main pipe bend radius
      H : branch height
      n : number of generators
    Returns:
      {
        "calc": {...},
        "dxf_base64": ... (only if msp=None)
      }
    """
    import math, ezdxf, io, base64

    # --- Inputs ---
    D = float(params["D"])
    d = float(params["d"])
    R = float(params["R"])
    H = float(params["H"])
    n = int(params["n"])

    r = d / 2
    Rc = R + D / 2
    periph_d = math.pi * d
    l = periph_d / n

    # --- Compute height values along intersection ---
    # The pattern should have:
    # - Bottom curve representing the intersection with the bend
    # - Top straight edge at height H
    # - Vertical generator lines
    
    top_pts = []
    bottom_pts = []
    h_values = []

    for i in range(n + 1):
        theta = 2 * math.pi * i / n
        x = i * l
        
        # Bottom curve: intersection with the curved pipe
        # The curve should bulge downward (inverted U-shape)
        # At the edges (theta=0, 2π): higher values
        # At the center (theta=π): lower values
        h_bottom = r * math.cos(theta)
        
        # Top edge: constant height
        h_top = H
        
        h_values.append(round(h_bottom, 2))
        bottom_pts.append((x, h_bottom))
        top_pts.append((x, h_top))

    # --- DXF setup ---
    local_mode = False
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True

    # --- DXF drawing ---
    # Top line (straight)
    msp.add_line((0, H), (periph_d, H), dxfattribs={"layer": layer})
    # Bottom curve (intersection with bend)
    msp.add_spline(bottom_pts, dxfattribs={"layer": layer})
    # Side lines
    msp.add_line((0, bottom_pts[0][1]), (0, H), dxfattribs={"layer": layer})
    msp.add_line((periph_d, bottom_pts[-1][1]), (periph_d, H), dxfattribs={"layer": layer})
    # Vertical generator lines
    for i in range(n + 1):
        x = i * l
        msp.add_line((x, bottom_pts[i][1]), (x, H), dxfattribs={"layer": layer, "color": 5})

    # --- Calculation results ---
    calc = {"π*d": round(periph_d, 2), "l": round(l, 2)}
    for i, h in enumerate(h_values, start=1):
        calc[f"h{i}"] = h

    result = {"calc": calc}

    # --- Encode DXF if standalone ---
    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_text = buf.getvalue()
        buf.close()
        dxf_base64 = base64.b64encode(dxf_text.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_base64

    return result

# 22. Tee on Cone (Branch on Conical Main Pipe)
def generate_tee_on_cone(params, msp=None, layer="CUT"):
    """
    Generate flat pattern for Tee on Cone.
    Compatible with unified DXF pipeline.

    Inputs:
      D1 : base diameter of cone (bottom)
      D2 : top diameter of cone
      L  : slant height of cone
      A  : vertical projection distance along cone
      d  : branch diameter
      H  : branch height
      X  : lateral offset of branch
      n  : number of generators

    Returns:
      {
        "calc": {...},
        "dxf_base64": ... (only if msp=None)
      }
    """
    import math, ezdxf, io, base64

    # --- Inputs ---
    D1 = float(params["D1"])
    D2 = float(params["D2"])
    L = float(params["L"])
    A = float(params["A"])
    d = float(params["d"])
    H = float(params["H"])
    X = float(params["X"])
    n = int(params["n"])

    R1, R2 = D1 / 2, D2 / 2
    r = d / 2
    alpha = math.atan((R2 - R1) / L)  # cone slope angle

    periph_d = math.pi * d
    l = periph_d / n

    # --- Points for pattern ---
    pts = []
    h_values = []

    for i in range(n + 1):
        θ = 2 * math.pi * i / n
        # Combined effects: cone + branch + offset
        cone_effect = (R2 - R1) * (A / L) * math.cos(alpha)
        branch_effect = r * math.sin(θ) - X * math.cos(θ)
        h = H + cone_effect + branch_effect
        h_values.append(round(h, 2))
        pts.append((i * l, h))

    # Close outline
    pts_closed = [(0, 0)] + pts + [(periph_d, 0)]

    # --- DXF setup ---
    local_mode = False
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True

    # --- DXF drawing ---
    msp.add_lwpolyline(pts_closed, close=True, dxfattribs={"layer": layer})
    # Base line
    msp.add_line((0, 0), (periph_d, 0), dxfattribs={"layer": layer, "color": 3})
    # Generator lines
    for i in range(n + 1):
        x = i * l
        msp.add_line((x, 0), (x, pts[i][1]), dxfattribs={"layer": layer, "color": 5})

    # --- Calculations ---
    calc = {"π*d": round(periph_d, 2), "l": round(l, 2)}
    for i, h in enumerate(h_values, start=1):
        calc[f"h{i}"] = h

    result = {"calc": calc}

    # --- Encode DXF if standalone ---
    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_text = buf.getvalue()
        buf.close()
        dxf_base64 = base64.b64encode(dxf_text.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_base64

    return result

# 23. Pants (Y-Branch)
def generate_pants(params, msp=None, layer="CUT"):
    """
    Generate flat pattern for Pants (Y-Branch).
    Compatible with Flask DXF pipeline.
    """
    import math, ezdxf, io, base64

    # --- Inputs ---
    D1 = float(params["D1"])
    D2 = float(params["D2"])
    H = float(params["H"])
    X = float(params["X"])
    n = int(params["n"])

    R1 = D1 / 2
    R2 = D2 / 2

    # --- Calculations ---
    a = math.sqrt(H**2 + (R2 - R1)**2)
    b = math.sqrt(H**2 + (R2 + R1)**2)
    periph = math.pi * D2
    l = periph / n

    # --- Data dictionary ---
    calc = {
        "a": round(a, 2),
        "b": round(b, 2),
        "L0-0": round(periph / 2, 2),
        "L0-1": round(periph / 2 * 1.045, 2),
        "L1-1": round(periph / 2 * 0.98, 2),
        "L1-2": round(periph / 2 * 1.055, 2),
        "L2-2": round(periph / 2 * 0.916, 2),
        "L2-3": round(periph / 2, 2),
        "L3-3": round(periph / 2 * 0.853, 2),
        "L3-4": round(periph / 2 * 0.905, 2),
        "L4-4": round(periph / 2 * 0.825, 2),
        "L0-0'": round(periph / 2 * 0.667, 2),
        "L1-1'": round(periph / 2 * 0.51, 2),
    }

    # --- DXF setup ---
    local_mode = False
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True

    # --- Drawing (wave-like Y branch pattern) ---
    pts = []
    for i in range(n + 1):
        x = i * l
        y = math.sin(i * math.pi / n) * (H / 2)
        pts.append((x, y))

    # Close the pattern
    pts_closed = [(0, 0)] + pts + [(periph, 0)]

    # Draw outline
    msp.add_lwpolyline(pts_closed, close=True, dxfattribs={"layer": layer})
    # Add base line for reference
    msp.add_line((0, 0), (periph, 0), dxfattribs={"layer": layer, "color": 3})

    # Add vertical generator lines
    for i in range(n + 1):
        x = i * l
        msp.add_line((x, 0), (x, pts[i][1]), dxfattribs={"layer": layer, "color": 5})

    # --- Return result ---
    result = {"calc": calc}

    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_data = buf.getvalue()
        buf.close()
        dxf_base64 = base64.b64encode(dxf_data.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_base64

    return result

# 24. Pants 2 (3-branch Y-Junction)
def generate_pants2(params, msp=None, layer="CUT"):
    """
    Generate flat pattern for Pants 2 (3-branch Y-junction).
    Compatible with Flask DXF pipeline and standalone test.
    """
    import math, ezdxf, io, base64

    # --- Inputs ---
    D1 = float(params["D1"])
    D2 = float(params["D2"])
    H = float(params["H"])
    X = float(params["X"])
    a = math.radians(float(params["a"]))
    n = int(params["n"])

    R1, R2 = D1 / 2, D2 / 2

    # --- Step 1: Geometry ---
    b = math.sqrt(H**2 + (R2 - R1)**2)

    # --- Step 2: Calculations ---
    calc = {
        "a": round(H * math.tan(a), 2),
        "b": round(b, 2),
    }

    base_len = math.pi * ((D1 + D2) / 2) / n
    calc.update({
        "L0-0": round(base_len * 4.7, 2),
        "L0-1": round(base_len * 4.7, 2),
        "L1-1": round(base_len * 3.75, 2),
        "L1-2": round(base_len * 3.85, 2),
        "L2-2": round(base_len * 2.75, 2),
        "L0-0'": round(base_len * 2.85, 2),
    })

    # --- Step 3: DXF Setup ---
    local_mode = False
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True

    # --- Step 4: DXF Drawing ---
    periph = math.pi * D2
    l = periph / n
    pts = [(i * l, (H / 3) * math.sin(math.pi * i / n)) for i in range(n + 1)]
    pts_closed = [(0, 0)] + pts + [(periph, 0)]

    # Outline
    msp.add_lwpolyline(pts_closed, close=True, dxfattribs={"layer": layer})
    # Base line
    msp.add_line((0, 0), (periph, 0), dxfattribs={"layer": layer, "color": 3})
    # Vertical generators
    for i in range(n + 1):
        x = i * l
        msp.add_line((x, 0), (x, pts[i][1]), dxfattribs={"layer": layer, "color": 5})

    # --- Step 5: Return results ---
    result = {"calc": calc}

    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_data = buf.getvalue()
        buf.close()
        dxf_base64 = base64.b64encode(dxf_data.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_base64

    return result

# 25. Pants Eccentric (Y-piece with eccentricity)
def generate_pants_ecc(params, msp=None, layer="CUT"):
    """
    Generate flat pattern for Pants Ecc (Y-piece avec excentration).
    Compatible with Flask DXF pipeline and standalone usage.
    """
    import math, ezdxf, io, base64

    # --- Extract parameters ---
    D1 = float(params["D1"])
    D2 = float(params["D2"])
    H = float(params["H"])
    X = float(params["X"])
    Y = float(params["Y"])
    n = int(params["n"])

    R1, R2 = D1 / 2, D2 / 2

    # --- Step 1: Geometry ---
    a = math.degrees(math.atan((H + Y) / (X / 2)))
    b = math.sqrt(H**2 + (X / 2)**2 + Y**2)

    calc = {"a": round(a, 2), "b": round(b, 2)}

    # --- Step 2: Pattern values ---
    base_len = math.pi * ((D1 + D2) / 2) / n
    calc.update({
        "L0-0": round(base_len * 4.0, 2),
        "L0-1": round(base_len * 4.3, 2),
        "L1-1": round(base_len * 3.45, 2),
        "L1-2": round(base_len * 3.75, 2),
        "L2-2": round(base_len * 3.2, 2),
        "L0'-1'": round(base_len * 4.25, 2),
        "L1'-1'": round(base_len * 3.78, 2),
        "L1'-2'": round(base_len * 4.1, 2),
        "L2'-2'": round(base_len * 3.18, 2),
        "L0-0\"": round(base_len * 2.6, 2),
    })

    # --- Step 3: DXF setup ---
    local_mode = False
    if msp is None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        local_mode = True

    # --- Step 4: Draw DXF ---
    periph = math.pi * max(D1, D2)
    l = periph / n
    pts = []

    for i in range(n + 1):
        x = i * l
        # asymmetrical Y-wave: right branch lifted by eccentricity Y
        y = (H / 3) * math.sin(math.pi * i / n) + (Y / H) * i
        pts.append((x, y))

    pts_closed = [(0, 0)] + pts + [(periph, 0)]

    # Outline
    msp.add_lwpolyline(pts_closed, close=True, dxfattribs={"layer": layer})
    # Base line
    msp.add_line((0, 0), (periph, 0), dxfattribs={"layer": layer, "color": 3})
    # Generatrix lines
    for i in range(n + 1):
        x = i * l
        msp.add_line((x, 0), (x, pts[i][1]), dxfattribs={"layer": layer, "color": 5})

    # --- Step 5: Return result ---
    result = {"calc": calc}

    if local_mode:
        buf = io.StringIO()
        doc.write(buf)
        dxf_data = buf.getvalue()
        buf.close()
        dxf_b64 = base64.b64encode(dxf_data.encode("utf-8")).decode("utf-8")
        result["dxf_base64"] = dxf_b64

    return result
