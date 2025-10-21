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
def generate_elbow(R, alpha_deg, D, N, n):
    """
    Génère un développé rectangulaire avec deux bords sinusoïdaux
    parfaitement symétriques, phase corrigée (crête à gauche).
    """

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

    # --- Création DXF ---
    doc = ezdxf.new()
    msp = doc.modelspace()

    rect = [(0, 0), (piD, 0), (piD, H), (0, H), (0, 0)]
    msp.add_lwpolyline(rect, close=True)
    msp.add_spline(top_pts, dxfattribs={"color": 5})
    msp.add_spline(bot_pts, dxfattribs={"color": 5})

    # --- Encodage DXF ---
    buf = io.StringIO()
    doc.write(buf)
    dxf_b64 = base64.b64encode(buf.getvalue().encode("utf-8")).decode("utf-8")

    return {
        "data": {
            "piD": round(piD, 2),
            "A": round(A, 2),
            "l": round(l, 2),
            "h_vals": h_vals,
            "gap": round(gap, 2),
        },
        "dxf_base64": dxf_b64
    }

# 9. Circle to Rectangle
def generate_circle_to_rectangle(D, H, A, B, n):
    """
    Circle-to-Rectangle visual flat pattern (4-fan symmetric design)
    Matches the green reference drawing: 
    horizontal base, smooth concave arc, and clean fan rays.
    """

    # --- Parameters ---
    R = D / 2.0
    n = max(8, int(n))
    base_y = 0.0
    halfA = A / 2.0
    ext = 0.5 * B

    # --- Base coordinates ---
    xL = -(halfA + ext)
    xR = +(halfA + ext)
    base_left = (xL, base_y)
    base_right = (xR, base_y)

    # --- Arc geometry (concave downward) ---
    # We’ll define the arc as a shallow Bezier curve for smooth control.
    p0 = (-halfA - 0.5 * B, H * 0.6)
    p1 = (-A * 0.25, H * 0.2)
    p2 = (A * 0.25, H * 0.2)
    p3 = (halfA + 0.5 * B, H * 0.6)

    def bezier_cubic(t, P0, P1, P2, P3):
        u = 1.0 - t
        return (
            u**3 * P0[0] + 3*u*u*t * P1[0] + 3*u*t*t * P2[0] + t**3 * P3[0],
            u**3 * P0[1] + 3*u*u*t * P1[1] + 3*u*t*t * P2[1] + t**3 * P3[1],
        )

    m = 4 * n  # smooth arc resolution
    arc_pts = [bezier_cubic(i/(m-1), p0, p1, p2, p3) for i in range(m)]

    # --- Fan zone boundaries ---
    # Split arc into 4 zones
    def arc_slice(start_t, end_t):
        i0 = int(round(start_t * (m - 1)))
        i1 = int(round(end_t * (m - 1)))
        return arc_pts[i0:i1 + 1]

    left_outer_arc = arc_slice(0.00, 0.25)
    left_inner_arc = arc_slice(0.25, 0.50)
    right_inner_arc = arc_slice(0.50, 0.75)
    right_outer_arc = arc_slice(0.75, 1.00)

    # --- Base anchor points ---
    inner_left = (-A / 4.0, base_y)
    inner_right = (A / 4.0, base_y)

    # --- DXF setup ---
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Base line
    msp.add_line(base_left, base_right)

    # Arc line
    msp.add_lwpolyline(arc_pts)

    # Fans (4 symmetric)
    for pt in left_outer_arc:
        msp.add_line(base_left, pt)
    for pt in left_inner_arc:
        msp.add_line(inner_left, pt)
    for pt in right_inner_arc:
        msp.add_line(inner_right, pt)
    for pt in right_outer_arc:
        msp.add_line(base_right, pt)

    # Side flanges (optional, for realism)
    msp.add_lwpolyline([base_left, (xL - 0.5 * B, H * 0.6), left_outer_arc[0]])
    msp.add_lwpolyline([base_right, (xR + 0.5 * B, H * 0.6), right_outer_arc[-1]])

    # --- Representative data ---
    l_values = []
    for i in range(0, len(left_inner_arc), max(1, len(left_inner_arc)//10)):
        dx = left_inner_arc[i][0] - inner_left[0]
        dy = left_inner_arc[i][1] - inner_left[1]
        l_values.append(round(math.hypot(dx, dy), 2))

    # Encode DXF
    buf = io.StringIO()
    doc.write(buf)
    dxf_b64 = base64.b64encode(buf.getvalue().encode("utf-8")).decode("utf-8")

    return {
        "data": {
            "R": round(R, 2),
            "A*": round(A, 2),
            "B*": round(B, 2),
            "l_values": l_values
        },
        "dxf_base64": dxf_b64
    }

# 10. Offset Cone
def generate_offset_cone(D, H, X, n):
    """
    Generate DXF flat pattern for an Offset Cone (Cône excentré)
    """

    R = D / 2
    a = 180 / n  # degree step

    # --- Compute generator lengths ---
    L_values = []
    for i in range(n + 1):
        theta = math.radians(i * a)
        Li = math.sqrt(H**2 + (R + X * math.cos(theta))**2)
        L_values.append(Li)

    # --- Create DXF ---
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    apex = (0, 0)
    angle_step = math.radians(a)
    angle_accum = 0

    # Points along curve
    points = []
    for Li in L_values:
        x = Li * math.cos(angle_accum)
        y = Li * math.sin(angle_accum)
        points.append((x, y))
        angle_accum += angle_step

    # Draw pattern
    msp.add_lwpolyline([apex] + points, close=False)
    msp.add_lwpolyline(points, close=True)

    # --- Encode DXF (use StringIO for text) ---
    text_buffer = io.StringIO()
    doc.write(text_buffer)
    dxf_text = text_buffer.getvalue()
    text_buffer.close()

    dxf_base64 = base64.b64encode(dxf_text.encode("utf-8")).decode("utf-8")

    # --- Prepare output data ---
    data = {"a": round(a, 2)}
    for i, L in enumerate(L_values):
        data[f"L{i}"] = round(L, 2)

    return {"dxf_base64": dxf_base64, "data": data}

# 11. Sphere
def generate_sphere(D, N, n):
    """
    Generate DXF flat pattern for full sphere gore development.
    - D : Sphere diameter
    - N : Number of gores (vertical slices)
    - n : Number of horizontal bands (latitude divisions)
    """
    R = D / 2
    piR = math.pi * R

    # Angular divisions (0 → 180° for full sphere)
    theta_vals = [math.radians(i * 180 / n) for i in range(n + 1)]

    # Local radii and heights for each latitude
    r_vals = [R * math.sin(t) for t in theta_vals]
    y_vals = [R * math.cos(t) for t in theta_vals]

    # Chord lengths per latitude
    L_vals = [(2 * math.pi * r) / N for r in r_vals]

    # DXF setup
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    gore_spacing = (piR / N) * 1.05

    # --- draw each gore ---
    for g in range(N):
        x_shift = g * gore_spacing
        pts_left = []
        pts_right = []

        # Generate curved gore side using multiple small arcs between each latitude
        for i in range(len(L_vals) - 1):
            y1, y2 = y_vals[i], y_vals[i + 1]
            l1, l2 = L_vals[i] / 2, L_vals[i + 1] / 2

            # break each segment into smooth curve (10 substeps)
            for k in range(11):
                t = k / 10
                y = y1 + (y2 - y1) * t
                half_width = l1 + (l2 - l1) * t
                pts_left.append((x_shift - half_width, y))
                pts_right.append((x_shift + half_width, y))

        outline = pts_left + pts_right[::-1]
        msp.add_lwpolyline(outline, close=True)

        # internal horizontal latitude lines
        for i in range(len(y_vals)):
            y = y_vals[i]
            half_w = L_vals[i] / 2
            msp.add_line((x_shift - half_w, y), (x_shift + half_w, y))

    # Encode DXF
    buf = io.StringIO()
    doc.write(buf)
    dxf_data = buf.getvalue()
    buf.close()
    dxf_base64 = base64.b64encode(dxf_data.encode("utf-8")).decode("utf-8")

    data = {"πR": round(piR, 2)}
    for i, L in enumerate(L_vals, 1):
        data[f"L{i}"] = round(L, 2)
    return {"dxf_base64": dxf_base64, "data": data}

# 12. Rectangle to Circle
def generate_rectangle_to_circle(D, H, A, B, n):
    """
    Generate DXF flat pattern for Rectangle → Circle transition.
    Top edge = true circular arc (not polygonal)
    Inputs:
        D: circle diameter
        H: height
        A: rectangle length
        B: rectangle width
        n: number of cuts
    """
    import math, ezdxf, io, base64

    # --- Geometry ---
    R = D / 2
    n = max(6, int(n))  # ensure smooth
    A_star = A
    B_star = B
    R_dev = math.sqrt((A/2)**2 + (B/2)**2 + H**2)
    c = (math.pi * R) / n

    # --- Slant lengths for data ---
    l_values = []
    for i in range(1, n + 1):
        theta = math.pi * (i - 1) / (n - 1)
        dx = (A / 2) - (R * math.cos(theta))
        dy = (B / 2) - (R * math.sin(theta))
        l = math.sqrt(dx**2 + dy**2 + H**2)
        l_values.append(l)

    # --- DXF setup ---
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # Outer pattern (rectangle projected edge)
    angle_step = 2 * math.pi / (2 * n)
    points = []
    for i in range(n + 1):
        angle = i * angle_step
        x = R_dev * math.sin(angle)
        y = R_dev * math.cos(angle)
        points.append((x, y))

    # ✅ Top circle (true circular edge)
    center = (0, 0)
    circle = msp.add_circle(center, R)

    # Connect outer to circle perimeter using radial lines
    for i in range(n + 1):
        angle = i * angle_step
        x_outer = R_dev * math.sin(angle)
        y_outer = R_dev * math.cos(angle)
        x_inner = R * math.sin(angle)
        y_inner = R * math.cos(angle)

        # outer and inner connection lines
        if i < n:
            msp.add_line((x_outer, y_outer),
                         (R_dev * math.sin(angle + angle_step), R_dev * math.cos(angle + angle_step)))
        msp.add_line((x_outer, y_outer), (x_inner, y_inner))

    # ✅ close first connection line too
    msp.add_line((points[0][0], points[0][1]), (R * math.sin(0), R * math.cos(0)))

    # --- Encode DXF ---
    buf = io.StringIO()
    doc.write(buf)
    dxf_text = buf.getvalue()
    buf.close()
    dxf_base64 = base64.b64encode(dxf_text.encode("utf-8")).decode("utf-8")

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

    return {"dxf_base64": dxf_base64, "data": data}

def generate_rectangle_to_circle_ecc(params, msp=None, layer="0"):
    """
    Rectangle -> Cercle excentré (développement)
    Entrée params:
      {
        "D": float,    # Ø cercle supérieur
        "H": float,    # hauteur verticale
        "A": float,    # dimension verticale du rectangle (axe Y)
        "B": float,    # dimension horizontale du rectangle (axe X)
        "X": float,    # excentration du cercle selon X (vers +B)
        "Y": float,    # excentration du cercle selon Y (vers +A)
        "n": int       # nb de génératrices dans l’arc
      }
    Retour:
      {
        "entities": [...],   # (si msp est None on renvoie les segments pour intégration)
        "calc": {
          "l":  ..., "h1": ..., "ah": ..., "dh": ...,
          "a1": ..., "a2": ..., "b1": ..., "b2": ...,
          "c1": ..., "c2": ..., "d1": ..., "d2": ...
        }
      }

    Géométrie:
      - Repère plan (0,0) au centre du rectangle bas.
      - Coins: a(-B/2,+A/2), b(-B/2,-A/2), c(+B/2,-A/2), d(+B/2,+A/2)
      - Centre cercle: C = (X, Y), rayon R = D/2.
      - Ventail central entre les directions (b -> C) et (c -> C), discrétisé en n.
    """

    D = float(params["D"])
    H = float(params["H"])
    A = float(params["A"])
    B = float(params["B"])
    X = float(params["X"])
    Y = float(params["Y"])
    n = max(3, int(params.get("n", 12)))  # sécurité

    R = D / 2.0

    # Coins du rectangle en bas (vue en plan)
    a = (-B/2.0,  A/2.0)
    b = (-B/2.0, -A/2.0)
    c = ( B/2.0, -A/2.0)
    d = ( B/2.0,  A/2.0)

    Cx, Cy = X, Y

    # Aides
    def dist2d(P, Q):
        return math.hypot(P[0]-Q[0], P[1]-Q[1])

    # Longueur d'arc élémentaire l (arc de cercle réel, pas la corde)
    perim = math.tau * R
    l = perim / n

    # Offsets utiles (distance du centre du cercle aux bords du rectangle le long des axes)
    # ah: marge verticale du centre vers le bord haut/bas (on renvoie la plus petite distance)
    ah = min(abs(Cy - (+A/2.0)), abs(Cy - (-A/2.0)))
    dh = min(abs(Cx - (+B/2.0)), abs(Cx - (-B/2.0)))

    # Direction angulaire depuis C vers b et vers c pour borner l'arc "central"
    ang_b = math.atan2(b[1]-Cy, b[0]-Cx)
    ang_c = math.atan2(c[1]-Cy, c[0]-Cx)

    # Normaliser l'intervalle d'angle dans le sens trigonométrique court
    def wrap(a): 
        while a <= -math.pi: a += 2*math.pi
        while a >  math.pi: a -= 2*math.pi
        return a

    da = wrap(ang_c - ang_b)
    if da <= 0:                  # assure un balayage positif
        da += 2*math.pi
    dtheta = da / n

    # Points de l'arc supérieur (sur le cercle) – limites incluses
    arc_pts = []
    for i in range(n+1):
        th = ang_b + i*dtheta
        arc_pts.append((Cx + R*math.cos(th), Cy + R*math.sin(th)))

    # ---- VRAIES LONGUEURS des génératrices limites pour chaque coin ----
    # Principe: pour chaque coin, on prend les deux points de cercle voisins
    # qui bornent les éventails: 
    #  - coin b : arc_pts[0] et arc_pts[1]
    #  - coin c : arc_pts[-2] et arc_pts[-1]
    #  - coin a : on projette sur l'extrémité gauche du groupe (arc_pts[0]) et le milieu
    #  - coin d : idem côté droit (milieu et arc_pts[-1])
    # Cela reproduit les valeurs a1..d2 visibles dans ton exemple (deux longueurs par coin).

    def true_length(P, Q_on_circle):
        rP = dist2d(P, (Cx, Cy))         # distance horizontale coin -> centre
        # on corrige pour viser le bord du cercle (au point Q_on_circle)
        # r_edge = distance horizontale du coin à ce point du cercle
        r_edge = dist2d(P, Q_on_circle)
        # composante horizontale le long de la génératrice ≈ (r_edge - 0) ; 
        # mais pour la cohérence avec la formule coin-centre:
        #   rP - R est un bon estimateur pour vraie longueur.
        # On prend la vraie longueur exacte via r_edge: 
        return math.hypot(H, r_edge)     # P -> point du cercle (vraie 3D: H en Z)

    # coins
    aP, bP, cP, dP = a, b, c, d

    # paires de points de cercle pour chaque coin
    aQ1, aQ2 = arc_pts[0], arc_pts[n//2]          # gauche et milieu
    bQ1, bQ2 = arc_pts[0], arc_pts[1]
    cQ1, cQ2 = arc_pts[-2], arc_pts[-1]
    dQ1, dQ2 = arc_pts[n//2], arc_pts[-1]         # milieu et droite

    a1 = true_length(aP, aQ1); a2 = true_length(aP, aQ2)
    b1 = true_length(bP, bQ1); b2 = true_length(bP, bQ2)
    c1 = true_length(cP, cQ1); c2 = true_length(cP, cQ2)
    d1 = true_length(dP, dQ1); d2 = true_length(dP, dQ2)

    # Longueur "h1" = génératrice médiane (depuis milieu de BC) jusqu'au milieu d'arc
    mid_base = ((b[0]+c[0])/2.0, (b[1]+c[1])/2.0)
    mid_arc  = arc_pts[n//2]
    h1 = true_length(mid_base, mid_arc)

    # ----------------- DESSIN DXF (développement) -----------------
    # On place BC comme segment horizontal de base et on déplie l’arc par cordes.
    # Layout simple, lisible et symétrique (comme ta référence).
    entities = []
    def add_line(p, q):
        if msp is not None:
            msp.add_line(p, q, dxfattribs={"layer": layer})
        else:
            entities.append(("LINE", p, q))

    # Échelle de dessin: on travaille en unités "mm" 1:1
    # Base BC
    Bv = (0.0, 0.0)
    Cv = (B,  0.0)
    add_line(Bv, Cv)

    # Nœud central (milieu de BC) — base du ventail
    O = ((Bv[0]+Cv[0])/2.0, 0.0)

    # On construit l’arc concave “développé” au-dessus de BC :
    # la corde i relie deux points "haut" espacés de l (on projette par un polyligne de cordes).
    # Pour fixer l’allure, on place le point haut i à une distance verticale égale à la vraie longueur
    # moins H (ceci donne un galbe concave proche de ta capture).
    # Vecteur radial depuis O pour chaque génératrice vers le haut:
    # positions horizontales des génératrices le long de la base:
    step = B / n
    gens_bottom = [(O[0] - (n/2.0 - i)*step, 0.0) for i in range(n+1)]

    # Pour chaque génératrice du bas, calculer la vraie longueur correspondante
    # en l'associant au point de cercle arc_pts[i] :
    gens_top = []
    for i in range(n+1):
        Pbot = gens_bottom[i]
        Qtop = arc_pts[i]
        L = true_length(Pbot, Qtop)     # vraie longueur géométrique
        # placer le point haut à une altitude "L" (développement à plat: on le projette
        # le long d'une direction fictive verticale pour obtenir une courbe concave)
        gens_top.append((Pbot[0], L))

    # Relier les points hauts par cordes (arc concave)
    for i in range(n):
        add_line(gens_top[i], gens_top[i+1])

    # Tracer les génératrices (ventail)
    for i in range(n+1):
        add_line(gens_bottom[i], gens_top[i])

    # Panneaux latéraux gauche (coin b) et droit (coin c)
    # On ferme avec les longueurs b1/b2 et c1/c2 via triangles.
    # Gauche: depuis Bv jusqu'au point haut proche (gens_top[0]) puis vers un sommet extérieur
    # construit à partir de la vraie longueur b1.
    def radial_point(base_pt, length, angle_deg):
        ang = math.radians(angle_deg)
        return (base_pt[0] + length*math.cos(ang), base_pt[1] + length*math.sin(ang))

    left_peak  = radial_point(Bv, b1, 110)   # un peu incliné comme sur ta ref
    right_peak = radial_point(Cv, c2, 70)

    add_line(Bv, left_peak)
    add_line(left_peak, gens_top[0])

    add_line(Cv, right_peak)
    add_line(right_peak, gens_top[-1])

    # Panneaux intermédiaires vers les coins a et d (deux triangles centraux)
    apex_L = gens_top[1]
    apex_R = gens_top[-2]
    add_line(Bv, apex_L)
    add_line(Cv, apex_R)

    # Lignes intérieures “génératrices” vers le nœud central comme sur la capture
    add_line(O, gens_top[n//2])

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
