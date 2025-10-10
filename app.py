from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import ezdxf
import io
import math
import os

# --- Import shapes ---
from shapes import (
    generate_cone,
    generate_frustum_cone,
    generate_frustum_cone_triangulation,
    generate_pyramid,
    generate_rectangle_to_rectangle,
    generate_flange,
    generate_truncated_cylinder,
    generate_bend
)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ Flat Pattern API is running"

@app.route("/generate_dxf", methods=["POST"])
def generate_dxf():
    try:
        data = request.get_json()
        shape = data.get("shape")

        if not shape:
            return jsonify({"error": "Shape not provided"}), 400

        # Always assume millimeters
        doc = ezdxf.new()
        msp = doc.modelspace()

        # --- Shape selection ---
        if shape == "cone":
            pts = generate_cone(float(data["diameter"]), float(data["height"]))
            msp.add_lwpolyline(pts, close=True)

        elif shape == "frustum_cone":
            pts = generate_frustum_cone(
                float(data["diameter1"]),
                float(data["diameter2"]),
                float(data["height"])
            )
            msp.add_lwpolyline(pts, close=True)

        elif shape == "frustum_cone_triangulation":
            pts = generate_frustum_cone_triangulation(
                float(data["diameter1"]),
                float(data["diameter2"]),
                float(data["height"])
            )
            msp.add_lwpolyline(pts, close=True)

        elif shape == "pyramid":
            pts = generate_pyramid(
                base=float(data["base"]),
                height=float(data["height"]),
                thickness=float(data.get("thickness", 2.0)),
                bend_radius=float(data.get("bend_radius", 2.0)),
                k_factor=float(data.get("k_factor", 0.33)),
                bend_angle=float(data.get("bend_angle", 90)),
                sides=int(data.get("sides", 4))
            )
            msp.add_lwpolyline(pts, close=True)

        elif shape == "rectangle_to_rectangle":
            pts = generate_rectangle_to_rectangle(
                float(data["w1"]),
                float(data["h1"]),
                float(data["w2"]),
                float(data["h2"]),
                float(data["height"])
            )
            msp.add_lwpolyline(pts, close=True)

        elif shape == "flange":
            outer_d = float(data["outer_d"])
            inner_d = float(data["inner_d"])
            holes = int(data["holes"])
            hole_d = float(data["hole_d"])

            pts = generate_flange(outer_d, inner_d, holes, hole_d)
            msp.add_lwpolyline(pts, close=True)

            # cercle intérieur
            msp.add_circle((0, 0), inner_d / 2)

            # perçages
            for i in range(holes):
                theta = 2 * math.pi * i / holes
                x = (outer_d / 2 - hole_d) * math.cos(theta)
                y = (outer_d / 2 - hole_d) * math.sin(theta)
                msp.add_circle((x, y), hole_d / 2)

        elif shape == "truncated_cylinder":
            pts = generate_truncated_cylinder(
                float(data["diameter"]),
                float(data["height"]),
                float(data["angle"])
            )
            msp.add_lwpolyline(pts, close=True)

        elif shape == "bend":
            patterns = generate_bend(
                float(data["diameter"]),
                float(data["bend_angle"]),
                float(data["radius"]),
                int(data.get("divisions", 12))
            )
            for pts in patterns:
                msp.add_lwpolyline(pts, close=True)

        else:
            return jsonify({"error": f"Shape '{shape}' not supported"}), 400

        # --- Sauvegarde DXF ---
        buffer = io.BytesIO()
        try:
            if hasattr(doc, "write_stream"):
                doc.write_stream(buffer)
            else:
                text_buffer = io.StringIO()
                doc.write(text_buffer)
                text_data = text_buffer.getvalue().encode("utf-8")
                buffer = io.BytesIO(text_data)
        except Exception as e:
            return jsonify({"error": f"DXF write failed: {str(e)}"}), 500

        buffer.seek(0)

        # --- Return DXF file ---
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"{shape}.dxf",
            mimetype="application/dxf"
        )

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
