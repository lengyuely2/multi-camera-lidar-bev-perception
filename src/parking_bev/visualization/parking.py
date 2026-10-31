"""Parking-focused BEV dashboard. Grid is a metric reference, not an inferred map."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..parking.assistance import CLASS_TEXT, SECTOR_TEXT, ParkingReport, VehicleGeometry
from ..parking.geometry import rectangle


def find_font():
    candidates = ["C:/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    return next((item for item in candidates if Path(item).is_file()), None)


class ParkingRenderer:
    def __init__(self, vehicle: VehicleGeometry, width=1280, height=860, font_path=None):
        self.vehicle, self.width, self.height = vehicle, width, height
        self.font_path = font_path or find_font()
        self.fonts = {size: ImageFont.truetype(self.font_path, size) if self.font_path else ImageFont.load_default()
                      for size in (13, 15, 17, 20, 23, 27, 32)}
        self.scale = min((height - 180) / 18, (width * 0.65 - 60) / 18)
        self.origin = np.array([width * 0.335, height * 0.535])

    def pixel(self, points):
        values = np.asarray(points)
        return np.stack((self.origin[0] - values[..., 1] * self.scale,
                         self.origin[1] - values[..., 0] * self.scale), axis=-1)

    def render(self, report: ParkingReport, objects, speed_mps=0.0, steering_rad=0.0, scenario=""):
        image = Image.new("RGB", (self.width, self.height), (13, 22, 34))
        draw = ImageDraw.Draw(image, "RGBA")
        color = {"stop": (255, 100, 95), "caution": (255, 190, 83), "watch": (100, 195, 255),
                 "no_conflict_detected": (122, 196, 216), "unknown": (180, 190, 205)}[report.level]
        status_text = {"stop": "停止当前泊车动作", "caution": "注意车身周边", "watch": "留意周边目标",
                       "no_conflict_detected": "未检测到轨迹冲突", "unknown": "信息不足 · 提示不可用"}[report.level]
        def text(position, value, size=17, fill=(215, 225, 235)):
            draw.text(position, value, font=self.fonts[size], fill=fill)
        def polygon(points, fill=None, outline=None, width=1):
            pixels = [tuple(item) for item in self.pixel(points)]
            draw.polygon(pixels, fill=fill)
            if outline:
                draw.line(pixels + [pixels[0]], fill=outline, width=width, joint="curve")
        def wrapped(value, left, top, maximum_width, size=20):
            row = ""
            for char in value:
                if self.fonts[size].getlength(row + char) > maximum_width:
                    text((left, top), row, size)
                    top += size + 9
                    row = ""
                row += char
            if row:
                text((left, top), row, size)
                top += size + 9
            return top
        text((32, 22), "泊车周边感知", 32, (240, 245, 250))
        profile_label = "示例车型" if self.vehicle.profile_name.startswith("example") else "车型参数"
        text((35, 67), f"{profile_label}  {self.vehicle.length_m:.2f} × {self.vehicle.width_m:.2f} m   ·   轴距 {self.vehicle.wheelbase_m:.2f} m", 15, (143, 162, 182))
        right = int(self.width * 0.68)
        text((right, 26), "PARKING ASSIST", 23, (132, 157, 180))
        draw.rounded_rectangle((24, 105, right - 22, self.height - 69), radius=16, fill=(20, 34, 48))
        for value in range(-8, 9):
            a, b = self.pixel([[value, -8], [value, 8]])
            c, d = self.pixel([[-8, value], [8, value]])
            draw.line([tuple(a), tuple(b)], fill=(38, 54, 68), width=1)
            draw.line([tuple(c), tuple(d)], fill=(38, 54, 68), width=1)
        for value in (2, 4, 6):
            point = self.pixel([value, 7.7])
            text(tuple(point), f"{value}m", 13, (104, 132, 151))
        if len(report.envelope_polygons):
            stride = max(1, len(report.envelope_polygons) // 12)
            for body in report.envelope_polygons[::stride][::-1]:
                polygon(body, fill=(*color, 10), outline=(*color, 95))
            centers = report.body_polygons.mean(axis=1)
            draw.line([tuple(item) for item in self.pixel(centers)], fill=(*color, 140), width=2)
        alert_by_id = {item.track_id: item for item in report.alerts}
        for item in objects:
            if np.linalg.norm(item.center_xy) > 9:
                continue
            alert = alert_by_id.get(item.track_id)
            object_color = (255, 117, 112) if alert and alert.level == "stop" else (255, 190, 83) if alert else (128, 178, 205)
            if report.status != "active":
                object_color = (100, 116, 130)
            polygon(rectangle(item.center_xy, *item.size_lw, item.yaw_rad), fill=(*object_color, 80), outline=(*object_color, 240), width=2)
            center = self.pixel(item.center_xy)
            text((center[0] + 9, center[1] - 14), CLASS_TEXT.get(item.class_name, "目标"), 15, object_color)
            if item.velocity_known and np.linalg.norm(item.velocity_xy) > 0.1:
                tip = self.pixel(item.center_xy + item.velocity_xy)
                draw.line([tuple(center), tuple(tip)], fill=object_color, width=3)
                draw.ellipse((tip[0]-3, tip[1]-3, tip[0]+3, tip[1]+3), fill=object_color)
        ego = rectangle([0, 0], self.vehicle.length_m, self.vehicle.width_m, 0.)
        polygon(ego, fill=(196, 215, 224, 255), outline=(243, 250, 255), width=2)
        polygon(rectangle([0.45, 0], self.vehicle.length_m * .48, self.vehicle.width_m * .77, 0), fill=(40, 65, 83, 255))
        nose = self.pixel([[self.vehicle.length_m / 2 - .1, 0], [self.vehicle.length_m / 2 - .5, .25], [self.vehicle.length_m / 2 - .5, -.25]])
        draw.polygon([tuple(item) for item in nose], fill=(81, 213, 232))
        for side in (-1, 1):
            light = self.pixel([-self.vehicle.length_m / 2 + .16, side * self.vehicle.width_m * .34])
            draw.ellipse((light[0]-4, light[1]-4, light[0]+4, light[1]+4), fill=(255, 104, 97))
        text((44, 125), "前方 ↑", 17, (160, 184, 202))
        if scenario:
            text((44, self.height - 103), scenario, 17, (157, 182, 199))
        panel_left = right
        draw.rounded_rectangle((panel_left, 104, self.width - 26, self.height - 69), radius=16, fill=(24, 37, 52))
        text((panel_left + 22, 125), status_text, 23, color)
        wrapped(report.advisory, panel_left + 22, 173, self.width - panel_left - 70, 20)
        direction = {"forward": "前进", "reverse": "倒车", "stationary": "静止", "unknown": "未知"}[report.direction]
        text((panel_left + 22, 338), f"{direction}   {abs(speed_mps) * 3.6:.1f} km/h", 23)
        text((panel_left + 22, 378), f"前轮转角 {np.degrees(steering_rad):+.1f}°", 17, (154, 179, 199))
        draw.line((panel_left + 22, 418, self.width - 47, 418), fill=(56, 74, 92))
        text((panel_left + 22, 437), "车身间距 · 米", 20)
        order = ("front_left", "front_right", "left", "right", "rear_left", "rear_right", "front", "rear")
        for index, sector in enumerate(order):
            gap = report.sector_gaps_m[sector]
            label = "—" if gap is None else f"{gap:.2f}"
            column, row = index % 2, index // 2
            left = panel_left + 22 + column * 158
            top = 480 + row * 40
            text((left, top), SECTOR_TEXT[sector], 15, (147, 171, 191))
            text((left + 62, top - 1), label, 17, color if gap is not None and gap < .8 else (215, 225, 235))
        text((panel_left + 22, 672), "数据时效", 17, (149, 174, 193))
        perception_age = "—" if report.perception_age_s is None else f"{report.perception_age_s:.2f}s"
        vehicle_age = "—" if report.vehicle_state_age_s is None else f"{report.vehicle_state_age_s:.2f}s"
        text((panel_left + 22, 705), f"感知 {perception_age}   车辆 {vehicle_age}", 17)
        text((33, self.height - 48), "示意网格非车位识别结果；仅评估检测目标，未验证空地、路沿及未知障碍。", 15, (140, 159, 177))
        return np.asarray(image)[..., ::-1].copy()
