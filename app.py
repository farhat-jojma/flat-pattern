from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import ezdxf
import io
import math
import os
import traceback
import sys
import base64

# --- Import shape generators ---
try:
    from shapes import (
        generate_cone,
        generate_frustum_cone,
        generate_frustum_cone_triangulation,
        generate_pyramid,
        generate_rectangle_to_rectangle,
        generate_flange,
        generate_truncated_cylinder,
        generate_elbow,
        generate_circle_to_rectangle,
        generate_offset_cone,
        generate_sphere,
        generate_rectangle_to_circle,
        generate_rectangle_to_circle_ecc,
        generate_frustum_ecc_angle,
        generate_frustum_ecc_paral,
        generate_auger
    )
except Exception as e:
    print("Error importing shapes:", e)
    traceback.print_exc()

sys.stdout.flush()

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Flat Pattern API running"})


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@app.route("/generate_dxf", methods=["POST"])
def generate_dxf():
    try:
        data = request.get_json()
        shape = data.get("shape")
        params = data.get("params", {})

        if not shape:
            return jsonify({"error": "Shape not provided"}), 400

        # Create DXF document
        doc = ezdxf.new()
        msp = doc.modelspace()
        result = {}
        response_data = {}  # ✅ add default empty response container

        # ---------------------- SHAPES ----------------------

        if shape == "cone":
            result = generate_cone(float(params["diameter"]), float(params["height"]))
            msp.add_lwpolyline(result["points"], close=True)

        elif shape == "frustum_cone":
            d1 = float(params["diameter1"])
            d2 = float(params["diameter2"])
            if "height" in params:
                value = float(params["height"])
                mode = "H"
            elif "beta" in params:
                value = float(params["beta"])
                mode = "B"
            else:
                return jsonify({"error": "Missing height or beta parameter"}), 400
            result = generate_frustum_cone(d1, d2, value, mode)
            msp.add_lwpolyline(result["points"], close=True)

        elif shape == "frustum_cone_triangulation":
            d1 = float(params["diameter1"])
            d2 = float(params["diameter2"])
            if "height" in params:
                value = float(params["height"])
                mode = "H"
            elif "beta" in params:
                value = float(params["beta"])
                mode = "B"
            else:
                return jsonify({"error": "Missing height or beta parameter"}), 400
            n = int(params.get("n", 12))
            result = generate_frustum_cone_triangulation(d1, d2, value, mode, n)
            msp.add_lwpolyline(result["points"], close=True)
            for line in result["generators"]:
                msp.add_line(line[0], line[1])

        elif shape == "pyramid":
            AA = float(params["AA"])
            AB = float(params["AB"])
            H = float(params["H"])
            result = generate_pyramid(AA, AB, H)
            for face in result["faces"]:
                msp.add_lwpolyline(face, close=True)
            response_data = result["data"]

        elif shape == "rectangle_to_rectangle":
            ab = float(params["ab"])
            bc = float(params["bc"])
            H = float(params["H"])
            AB = float(params["AB"])
            BC = float(params["BC"])
            result = generate_rectangle_to_rectangle(ab, bc, H, AB, BC)
            for face in result["faces"]:
                msp.add_lwpolyline(face, close=True)
            response_data = result["data"]

        elif shape == "flange":
            result = generate_flange(
                float(params["D1"]),
                float(params["D2"]),
                float(params["D3"]),
                float(params["D4"]),
                int(params["N1"]),
                float(params["d1"]),
                int(params["N2"]),
                float(params["d2"]),
            )
            for etype, center, radius in result["entities"]:
                x, y = center
                if etype in ("cut", "hole"):
                    steps = 90
                    pts = [(x + radius * math.cos(2 * math.pi * i / steps),
                            y + radius * math.sin(2 * math.pi * i / steps))
                           for i in range(steps + 1)]
                    msp.add_lwpolyline(pts, close=True)
            response_data = result["data"]

        elif shape == "truncated_cylinder":
            result = generate_truncated_cylinder(
                float(params["diameter"]),
                float(params["height"]),
                float(params["angle"]),
                int(params["n"])
            )
            return jsonify(result)

        elif shape == "elbow":
            R = float(params["R"])
            alpha = float(params["alpha"])
            D = float(params["D"])
            N = int(params["N"])
            n = int(params["n"])
            result = generate_elbow(R, alpha, D, N, n)
            return jsonify(result)

        elif shape == "circle_to_rectangle":
            D = float(params["D"])
            H = float(params["H"])
            A = float(params["A"])
            B = float(params["B"])
            n = int(params["n"])
            result = generate_circle_to_rectangle(D, H, A, B, n)
            return jsonify(result)

        elif shape == "offset_cone":
            result = generate_offset_cone(
                float(params["D"]),
                float(params["H"]),
                float(params["X"]),
                int(params["n"])
            )
            return jsonify(result)

        elif shape == "sphere":
            result = generate_sphere(
                float(params["D"]),
                int(params["N"]),
                int(params["n"])
            )
            return jsonify(result)

        elif shape == "rectangle_to_circle":
            D = float(params["D"])
            H = float(params["H"])
            A = float(params["A"])
            B = float(params["B"])
            n = int(params["n"])
            result = generate_rectangle_to_circle(D, H, A, B, n)
            return jsonify(result)

        elif shape == "rectangle_to_circle_ecc":
            out = generate_rectangle_to_circle_ecc(params, msp=doc.modelspace(), layer="CUT")
            calc = out["calc"]
            response_data = calc  # ✅ now we attach the calc dict to response_data
            
        elif shape == "frustum_ecc_angle":
            out = generate_frustum_ecc_angle(params, msp=doc.modelspace(), layer="CUT")
            response_data = out["calc"]

        elif shape == "frustum_ecc_paral":
            out = generate_frustum_ecc_paral(params, msp=doc.modelspace(), layer="CUT")
            response_data = out["calc"]

        elif shape == "auger":
            out = generate_auger(params, msp=doc.modelspace(), layer="CUT")
            response_data = out["calc"]

        else:
            return jsonify({"error": f"Shape '{shape}' not supported"}), 400

        # ---------------------- DXF EXPORT ----------------------
        try:
            text_buffer = io.StringIO()
            doc.write(text_buffer)
            dxf_data = text_buffer.getvalue().encode("utf-8")
            dxf_base64 = base64.b64encode(dxf_data).decode("utf-8")
        except Exception as e:
            print("Error writing DXF:", e)
            traceback.print_exc()
            return jsonify({"error": f"DXF write failed: {str(e)}"}), 500

        # ✅ Unified JSON response with data + DXF
        response_json = {
            "shape": shape,
            "dxf_base64": dxf_base64,
            "data": response_data  # ✅ ensures calculated values appear in test.html
        }

        return jsonify(response_json)

    except Exception as e:
        print("Error:", e, file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
