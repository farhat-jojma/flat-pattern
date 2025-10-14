import math

# 1. Cone
def generate_cone(diameter, height):
    r = math.sqrt((diameter / 2) ** 2 + height ** 2)
    arc_length = math.pi * diameter
    angle = 360 * arc_length / (2 * math.pi * r)

    points = []
    steps = 100
    for i in range(steps + 1):
        theta = math.radians(i * angle / steps)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append((x, y))
    points.append((0, 0))
    return points


# 2. Frustum Cone
def generate_frustum_cone(d1, d2, height):
    r1 = math.sqrt((d1 / 2) ** 2 + height ** 2)
    r2 = math.sqrt((d2 / 2) ** 2 + height ** 2)
    arc_length = math.pi * (d1 + d2)
    angle = 360 * arc_length / (2 * math.pi * r2)

    points_outer = []
    points_inner = []
    steps = 100
    for i in range(steps + 1):
        theta = math.radians(i * angle / steps)
        points_outer.append((r2 * math.cos(theta), r2 * math.sin(theta)))
        points_inner.append((r1 * math.cos(theta), r1 * math.sin(theta)))
    return points_outer + list(reversed(points_inner))


# 3. Frustum Cone (Triangulation)
def generate_frustum_cone_triangulation(d1, d2, height, n=24):
    r1 = d1 / 2
    r2 = d2 / 2
    step = 2 * math.pi / n
    points_top = []
    points_bottom = []
    for i in range(n + 1):
        theta = i * step
        points_top.append((r1 * math.cos(theta), r1 * math.sin(theta)))
        points_bottom.append((r2 * math.cos(theta), r2 * math.sin(theta)))
    return points_top + list(reversed(points_bottom))


# 4. Pyramid (sheet metal version, supports K-factor & bend allowance)
def generate_pyramid(AA, AB, H):
    """
    Génère le développé d'une pyramide à base rectangulaire.
    AA = longueur du côté de base (en mm)
    AB = largeur du côté de base (en mm)
    H = hauteur verticale (en mm)
    Retourne une liste de polygones (1 par face).
    """
    # Points de la base dans le plan XY
    base_points = [
        (0, 0),          # A
        (AA, 0),         # B
        (AA, AB),        # B'
        (0, AB)          # A'
    ]

    # Centre de la base (C)
    Cx = AA / 2
    Cy = AB / 2

    # Calcul des longueurs inclinées (slant heights)
    slant1 = math.sqrt((AA / 2) ** 2 + H ** 2)  # côté parallèle à AA
    slant2 = math.sqrt((AB / 2) ** 2 + H ** 2)  # côté parallèle à AB

    # Génération des faces (triangles)
    faces = []

    # Face avant
    faces.append([(0, 0), (AA, 0), (Cx, -slant1)])
    # Face droite
    faces.append([(AA, 0), (AA, AB), (AA + slant2, Cy)])
    # Face arrière
    faces.append([(0, AB), (AA, AB), (Cx, AB + slant1)])
    # Face gauche
    faces.append([(0, 0), (0, AB), (-slant2, Cy)])

    return faces
# 5. Rectangle to Rectangle
def generate_rectangle_to_rectangle(w1, h1, w2, h2, height):
    return [
        (0, 0), (w1, 0), (w1, h1), (0, h1), (0, 0),
        (0, height), (w2, height), (w2, h2 + height), (0, h2 + height), (0, height)
    ]


# 6. Flange
def generate_flange(outer_d, inner_d, holes, hole_d):
    points = []
    steps = 100
    for i in range(steps):
        theta = 2 * math.pi * i / steps
        points.append((outer_d/2 * math.cos(theta), outer_d/2 * math.sin(theta)))
    points.append(points[0])
    return points


# 7. Truncated Cylinder
def generate_truncated_cylinder(diameter, height, angle):
    width = math.pi * diameter
    offset = height * math.tan(math.radians(angle))
    return [
        (0, 0),
        (width, 0),
        (width, height + offset),
        (0, height),
        (0, 0)
    ]

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
