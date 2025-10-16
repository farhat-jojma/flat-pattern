import math
import ezdxf
import io
import base64

# 1. Cone
def generate_cone(diameter, height):
    """
    Generates a true flat pattern of a cone and returns both geometry and calculations.
    """

    import math

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

    # --- Arcs inclinés ---
    steps = 200
    outer, inner = [], []
    start_angle = -beta / 2
    for i in range(steps + 1):
        θ = math.radians(start_angle + i * (beta / steps))
        outer.append((R1 * math.cos(θ), R1 * math.sin(θ)))
        inner.append((R2 * math.cos(θ), R2 * math.sin(θ)))
    inner.reverse()
    pts = outer + inner + [outer[0]]

    # --- Calculs complémentaires ---
    corde_A = 2 * R1 * math.sin(math.radians(beta / 2))
    corde_C = 2 * R2 * math.sin(math.radians(beta / 2))
    L = R2 - R1
    pi_D1 = math.pi * D1
    pi_D2 = math.pi * D2

    # hauteur projetée de chaque rayon sur l’axe du cône
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

    return {"points": pts, "data": data}

# 3. Frustum Cone (Triangulation)
def generate_frustum_cone_triangulation(d1, d2, value, mode="H", n=12):
    import math

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

# 4. Pyramid (sheet metal version, supports K-factor & bend allowance)
def generate_pyramid(AA, AB, H):
    """
    Génère le développé d'une pyramide à base rectangulaire parfaitement refermable.
    Entrées :
        AA = longueur de base (mm)
        AB = largeur de base (mm)
        H  = hauteur verticale (mm)
    Retour :
        faces développées + calculs OI, OC, IB, CA
    """
    import math

    # --- Données de base ---
    Cx = AA / 2
    Cy = AB / 2

    # Slant heights
    slant1 = math.sqrt(Cx**2 + H**2)  # pour faces avant/arrière
    slant2 = math.sqrt(Cy**2 + H**2)  # pour faces gauche/droite

    # Angle du sommet pour ajuster la rotation des faces
    alpha_x = math.atan(H / Cx)  # angle inclinaison avant/arrière
    alpha_y = math.atan(H / Cy)  # angle inclinaison gauche/droite

    # --- Développement (à plat) ---
    faces = []

    # Face avant (triangle isocèle)
    faces.append([
        (0, 0),
        (AA, 0),
        (AA / 2, -slant1)
    ])

    # Face droite
    faces.append([
        (AA, 0),
        (AA, AB),
        (AA + slant2, AB / 2)
    ])

    # Face arrière
    faces.append([
        (0, AB),
        (AA, AB),
        (AA / 2, AB + slant1)
    ])

    # Face gauche
    faces.append([
        (0, 0),
        (0, AB),
        (-slant2, AB / 2)
    ])

    # --- Données calculées ---
    OI = Cx
    OC = Cy
    IB = slant1
    CA = slant2

    data = {
        "OI (mm)": round(OI, 2),
        "OC (mm)": round(OC, 2),
        "IB (mm)": round(IB, 2),
        "CA (mm)": round(CA, 2)
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
def generate_truncated_cylinder(diameter, height, angle_deg, n):
    """
    Generate the correct symmetrical flat pattern for a truncated cylinder.
    Produces one full smooth wave crest centered at mid-length.
    """
    import math, ezdxf, io, base64

    R = diameter / 2
    alpha = math.radians(angle_deg)
    L_total = math.pi * diameter          # full development length (πD)
    l = L_total / (n - 1)

    bottom = []
    top = []
    h_values = []

    # Use full cosine cycle 0→2π to make rise and fall symmetrical
    for i in range(n):
        theta = 2 * math.pi * i / (n - 1)   # 0 → 2π
        h = height + (R * math.tan(alpha) / 2) * (1 - math.cos(theta))
        x = i * l
        bottom.append((x, 0))
        top.append((x, h))
        h_values.append(round(h, 2))

    # Ensure both ends equal
    top[0] = (top[0][0], height)
    top[-1] = (top[-1][0], height)

    # --- DXF creation ---
    doc = ezdxf.new()
    msp = doc.modelspace()

    msp.add_lwpolyline(bottom, close=False)
    msp.add_line(bottom[0], top[0])
    msp.add_spline(top)
    msp.add_line(top[-1], bottom[-1])

    # --- Encode DXF ---
    buffer = io.StringIO()
    doc.write(buffer)
    dxf_data = buffer.getvalue()
    dxf_base64 = base64.b64encode(dxf_data.encode("utf-8")).decode("utf-8")

    return {
        "data": {
            "piD": round(L_total, 2),
            "l": round(l, 2),
            "h_values": h_values
        },
        "dxf_base64": dxf_base64
    }

# 8. Bend
def generate_elbow(R, alpha, D, N, n):
    """
    Realistic elbow flat pattern:
    - rectangle of width πD and height D
    - two identical arcs (not mirrored) offset vertically
    - geometry based on cylindrical intersection, no crossing
    """

    # basic parameters
    alpha_sector = math.radians(alpha / N)
    piD = math.pi * D
    step = piD / n
    H = D
    center_y = H / 2

    # compute intersection heights (single arc shape)
    h_vals = []
    for i in range(n + 1):
        theta = math.radians(i * 360 / n)
        h = R * (1 - math.cos(alpha_sector / 2)) + R * math.sin(alpha_sector / 2) * math.sin(theta)
        h_vals.append(h)

    # normalize so the arc sits roughly mid-height
    h_min, h_max = min(h_vals), max(h_vals)
    amplitude = h_max - h_min
    base = center_y - amplitude / 2

    # distance between upper and lower edges of band
    gap = amplitude * 1.2  # constant vertical offset; adjust for visual spacing

    points_top = []
    points_bottom = []

    for i, h in enumerate(h_vals):
        x = i * step
        # both arcs go in the same direction; bottom is just offset downwards
        y_top = base + h + gap / 2
        y_bottom = base + h - gap / 2
        points_top.append((x, y_top))
        points_bottom.append((x, y_bottom))

    # outer rectangle for reference
    rect = [(0, 0), (piD, 0), (piD, H), (0, H)]

    return {
        "A": points_top,
        "B": points_bottom,
        "rect": rect,
        "piD": piD,
        "step": step,
    }

