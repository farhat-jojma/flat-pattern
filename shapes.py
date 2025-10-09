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
def generate_pyramid(base, height, thickness=2.0, bend_radius=2.0, k_factor=0.33, bend_angle=90, sides=4):
    """
    Génère le développé d'une pyramide polygonale (par défaut carrée) avec prise en compte
    des paramètres de pliage (épaisseur, rayon, K-factor).
    
    :param base: longueur du côté de la base (mm)
    :param height: hauteur verticale (mm)
    :param thickness: épaisseur matière (mm)
    :param bend_radius: rayon intérieur du pli (mm)
    :param k_factor: facteur neutre (0.3 ~ 0.5)
    :param bend_angle: angle de pliage en degrés (souvent 90°)
    :param sides: nombre de côtés (par défaut 4 = pyramide carrée)
    :return: liste de points XY pour DXF
    """
    # Apothem de la base (du centre à la moitié d'un côté)
    apothem = base / (2 * math.tan(math.pi / sides))

    # Slant height (génératrice) depuis le neutre
    slant_height = math.sqrt(apothem**2 + height**2)

    # Bend allowance pour un pli
    bend_allowance = math.radians(bend_angle) * (bend_radius + k_factor * thickness)

    # Longueur développée d'un côté = slant height + bend compensation
    true_length = slant_height + bend_allowance

    # Calcul de l'angle central
    central_angle = 2 * math.pi / sides

    points = []
    for i in range(sides + 1):
        theta = i * central_angle
        x = true_length * math.cos(theta)
        y = true_length * math.sin(theta)
        points.append((x, y))

    return points

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
def generate_bend(diameter, bend_angle, radius, divisions=12):
    """
    Génère le développé d'un coude (bend / elbow) par triangulation.
    :param diameter: diamètre du tube (mm)
    :param bend_angle: angle du coude (°)
    :param radius: rayon de cintrage (mm)
    :param divisions: nombre de génératrices (plus grand = plus précis)
    :return: liste de listes de points (chaque trapèze est une polyline)
    """
    r = diameter / 2
    angle_step = math.radians(bend_angle) / divisions

    # Liste des polylignes (trapèzes juxtaposés)
    patterns = []

    for i in range(divisions):
        theta1 = i * angle_step
        theta2 = (i + 1) * angle_step

        # Longueur développée pour chaque génératrice
        arc_length1 = (radius - r) * theta1
        arc_length2 = (radius + r) * theta1
        arc_length3 = (radius - r) * theta2
        arc_length4 = (radius + r) * theta2

        # Trapèze (points à plat)
        trapezoid = [
            (arc_length1, 0),
            (arc_length2, 0),
            (arc_length4, r * 2),
            (arc_length3, r * 2),
            (arc_length1, 0)
        ]
        patterns.append(trapezoid)

    return patterns