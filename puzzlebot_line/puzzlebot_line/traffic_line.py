#!/usr/bin/env python3

# ============================================================
# imports
# ============================================================

import math

import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from yolo_msg.msg import Yolov8Inference


# ============================================================
# nodo principal
# ============================================================

class TrafficLine(Node):

    def __init__(self):
        super().__init__('traffic_line')

        # puente para convertir entre imágenes de ros y opencv
        self.bridge = CvBridge()
        self.image_received = False
        self.cv_img = None

        # suscripción a la cámara del robot
        self.sub_camera = self.create_subscription(
            Image,
            '/video_source/raw',
            self.camera_callback,
            10
        )

        # suscripción a las detecciones de yolo
        self.sub_yolo = self.create_subscription(
            Yolov8Inference,
            '/Yolov8_inference',
            self.yolo_callback,
            10
        )

        # publicación de la imagen procesada para poder verla/debuggear
        self.pub_processed = self.create_publisher(
            Image,
            '/traffic_line/processed_img',
            10
        )

        # publicación del comando de velocidad del robot
        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # velocidades base del seguidor
        self.v_normal = 0.10
        self.v_slow = 0.07

        # ganancias y límites para corregir la dirección
        self.angular_gain = 1.2
        self.max_angular_speed = 1.2
        self.search_angular_speed = 0.0

        # parámetros hsv para detectar la línea negra
        self.declare_parameter('black_h_low', 0)
        self.declare_parameter('black_s_low', 0)
        self.declare_parameter('black_v_low', 0)
        self.declare_parameter('black_h_high', 180)
        self.declare_parameter('black_s_high', 255)
        self.declare_parameter('black_v_high', 110)

        # parámetros de región de interés y área mínima de línea
        self.declare_parameter('roi_start_ratio', 0.60)
        self.declare_parameter('min_line_area', 300)

        self.black_low = np.array([0, 0, 0])
        self.black_high = np.array([180, 255, 110])

        # parámetros para filtrar detecciones de yolo
        self.declare_parameter('min_detection_area', 280)
        self.declare_parameter('traffic_light_min_area', 150)
        self.declare_parameter('sign_cooldown', 15.0)

        # parámetros de velocidad para maniobras por tiempo
        self.declare_parameter('maneuver_linear_speed', 0.11)
        self.declare_parameter('maneuver_angular_speed', 0.8)

        # tiempos de avance y giro para las señales de dirección
        self.declare_parameter('straight_signal_duration', 6.0)
        self.declare_parameter('turn_forward_before_duration', 4.2)
        self.declare_parameter('turn_forward_after_duration', 1.0)
        self.declare_parameter('turn_compensation', 0.95)

        # tiempo que se conserva el último estado del semáforo
        self.declare_parameter('traffic_light_persist_time', 1.0)

        # valores iniciales de detección
        self.min_detection_area = 280.0
        self.traffic_light_min_area = 150.0
        self.sign_cooldown = 15.0

        # valores iniciales de maniobras
        self.maneuver_linear_speed = 0.13
        self.maneuver_angular_speed = 0.8

        self.straight_signal_duration = 6.0
        self.turn_forward_before_duration = 4.2
        self.turn_forward_after_duration = 1.0
        self.turn_compensation = 1.0
        self.turn_duration = (math.pi / 2.0) / self.maneuver_angular_speed * self.turn_compensation

        # memoria de detecciones y estados de yolo
        self.latest_detections = []
        self.yolo_msg_time = 0.0

        self.traffic_light_state = None
        self.waiting_for_green = False
        self.last_traffic_light_time = 0.0
        self.traffic_light_persist_time = 1.0

        # timers para señales que tienen efecto temporal
        self.stop_timer_end = 0.0
        self.construction_timer_end = 0.0
        self.give_way_timer_end = 0.0

        # estado de las maniobras guardadas o en ejecución
        self.pending_maneuver = None
        self.executing_maneuver = None
        self.maneuver_step = None
        self.maneuver_step_start_time = 0.0

        # pausa de maniobra cuando aparece rojo
        self.maneuver_paused = False
        self.maneuver_pause_start_time = 0.0

        # registro para evitar disparar la misma señal muchas veces seguidas
        self.last_triggered = {}

        # ciclo principal de control
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('traffic_line node started')
        self.get_logger().info(
            f'Timed maneuvers: straight={self.straight_signal_duration:.2f}s, '
            f'turn forward before={self.turn_forward_before_duration:.2f}s, '
            f'turn={self.turn_duration:.2f}s, '
            f'turn forward after={self.turn_forward_after_duration:.2f}s'
        )

    # ========================================================
    # callbacks de entrada
    # ========================================================

    def yolo_callback(self, msg):
        # guarda la última lista de detecciones que llegó de yolo
        self.latest_detections = list(msg.yolov8_inference)
        self.yolo_msg_time = self.get_clock().now().nanoseconds / 1e9

        for det in self.latest_detections:
            area = abs(det.bottom - det.top) * abs(det.right - det.left)

            self.get_logger().info(
                f'YOLO -> {det.class_name} area={area}'
            )

    def camera_callback(self, msg):
        # convierte la imagen de ros a formato opencv
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.cv_img = img
            self.image_received = True
        except Exception as e:
            self.get_logger().error('Error converting image: {}'.format(e))

    # ========================================================
    # lectura y protección de parámetros
    # ========================================================

    def get_int_param(self, name):
        return int(self.get_parameter(name).value)

    def get_float_param(self, name):
        return float(self.get_parameter(name).value)

    def clamp_h(self, value):
        return max(0, min(180, int(value)))

    def clamp_sv(self, value):
        return max(0, min(255, int(value)))

    def update_params(self):
        # actualiza los rangos hsv desde parámetros de ros
        self.black_low = np.array([
            self.clamp_h(self.get_int_param('black_h_low')),
            self.clamp_sv(self.get_int_param('black_s_low')),
            self.clamp_sv(self.get_int_param('black_v_low'))
        ])

        self.black_high = np.array([
            self.clamp_h(self.get_int_param('black_h_high')),
            self.clamp_sv(self.get_int_param('black_s_high')),
            self.clamp_sv(self.get_int_param('black_v_high'))
        ])

        # actualiza filtros de área y cooldown de señales
        self.min_detection_area = float(
            self.get_parameter('min_detection_area').value
        )

        self.traffic_light_min_area = float(
            self.get_parameter('traffic_light_min_area').value
        )

        self.sign_cooldown = float(
            self.get_parameter('sign_cooldown').value
        )

        # actualiza velocidades y tiempos de maniobra
        self.maneuver_linear_speed = float(
            self.get_parameter('maneuver_linear_speed').value
        )

        self.maneuver_angular_speed = float(
            self.get_parameter('maneuver_angular_speed').value
        )

        self.straight_signal_duration = float(
            self.get_parameter('straight_signal_duration').value
        )

        self.turn_forward_before_duration = float(
            self.get_parameter('turn_forward_before_duration').value
        )

        self.turn_forward_after_duration = float(
            self.get_parameter('turn_forward_after_duration').value
        )

        self.turn_compensation = float(
            self.get_parameter('turn_compensation').value
        )

        self.traffic_light_persist_time = float(
            self.get_parameter('traffic_light_persist_time').value
        )

        # recalcula cuánto debe durar un giro de 90 grados
        if abs(self.maneuver_angular_speed) > 1e-6:
            self.turn_duration = (
                (math.pi / 2.0) /
                abs(self.maneuver_angular_speed) *
                self.turn_compensation
            )
        else:
            self.turn_duration = 0.0

    # ========================================================
    # procesamiento de máscara y contornos
    # ========================================================

    def clean_mask(self, mask):
        # limpia ruido pequeño de la máscara
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        return mask

    def find_contours(self, mask):
        # compatibilidad para versiones distintas de opencv
        result = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(result) == 3:
            _, contours, _ = result
        else:
            contours, _ = result

        return contours

    def get_largest_contour(self, mask):
        # devuelve el contorno más grande de una máscara
        contours = self.find_contours(mask)

        if not contours:
            return None, 0.0

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        return contour, area

    def get_centroid(self, contour):
        # calcula el centro del contorno usando momentos
        M = cv2.moments(contour)

        if M['m00'] == 0:
            return 0, 0, False

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return cx, cy, True

    def detect_line_center(self, hsv_roi):
        # busca la línea negra dentro de la región inferior de la imagen
        mask_line = cv2.inRange(
            hsv_roi,
            self.black_low,
            self.black_high
        )

        mask_line = self.clean_mask(mask_line)

        contours = self.find_contours(mask_line)

        if not contours:
            return False, mask_line, 0, 0, 0.0, None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area < self.get_int_param('min_line_area'):
            return False, mask_line, 0, 0, area, contour

        cx, cy, valid_centroid = self.get_centroid(contour)

        if not valid_centroid:
            return False, mask_line, 0, 0, area, contour

        return True, mask_line, cx, cy, area, contour

    # ========================================================
    # maniobras por tiempo
    # ========================================================

    def start_timed_maneuver(self, maneuver):
        # arranca una maniobra guardada por yolo
        self.executing_maneuver = maneuver
        self.pending_maneuver = None
        self.maneuver_step_start_time = self.get_clock().now().nanoseconds / 1e9
        self.maneuver_paused = False

        if maneuver == 'STRAIGHT':
            self.maneuver_step = 'STRAIGHT_FORWARD'
        elif maneuver in ['LEFT', 'RIGHT']:
            self.maneuver_step = 'FORWARD_BEFORE_TURN'
        else:
            self.executing_maneuver = None
            self.maneuver_step = None
            return

        self.get_logger().warn(
            f'STARTING TIMED MANEUVER -> {self.executing_maneuver}'
        )

    def finish_timed_maneuver(self):
        # limpia el estado cuando termina una maniobra
        self.get_logger().warn(
            f'TIMED MANEUVER COMPLETE -> {self.executing_maneuver}'
        )

        self.executing_maneuver = None
        self.maneuver_step = None
        self.maneuver_paused = False

    def run_timed_maneuver(self, cmd, now):
        # ejecuta el paso actual de la maniobra
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        if self.executing_maneuver is None or self.maneuver_step is None:
            return

        elapsed = now - self.maneuver_step_start_time

        if self.maneuver_step == 'STRAIGHT_FORWARD':
            cmd.linear.x = self.maneuver_linear_speed
            cmd.angular.z = 0.0

            if elapsed >= self.straight_signal_duration:
                self.finish_timed_maneuver()

            return

        if self.maneuver_step == 'FORWARD_BEFORE_TURN':
            cmd.linear.x = self.maneuver_linear_speed
            cmd.angular.z = 0.0

            if elapsed >= self.turn_forward_before_duration:
                self.maneuver_step = 'TURN_90'
                self.maneuver_step_start_time = now
                self.get_logger().info('Timed step: TURN_90')

            return

        if self.maneuver_step == 'TURN_90':
            cmd.linear.x = 0.0

            if self.executing_maneuver == 'RIGHT':
                cmd.angular.z = -self.maneuver_angular_speed
            elif self.executing_maneuver == 'LEFT':
                cmd.angular.z = self.maneuver_angular_speed

            if elapsed >= self.turn_duration:
                self.maneuver_step = 'FORWARD_AFTER_TURN'
                self.maneuver_step_start_time = now
                self.get_logger().info('Timed step: FORWARD_AFTER_TURN')

            return

        if self.maneuver_step == 'FORWARD_AFTER_TURN':
            cmd.linear.x = self.maneuver_linear_speed
            cmd.angular.z = 0.0

            if elapsed >= self.turn_forward_after_duration:
                self.finish_timed_maneuver()

            return

    # ========================================================
    # ciclo principal de control
    # ========================================================

    def control_loop(self):
        # si todavía no hay imagen, no se puede controlar nada
        if not self.image_received or self.cv_img is None:
            return

        self.update_params()

        img = self.cv_img.copy()
        height, width, _ = img.shape

        now = self.get_clock().now().nanoseconds / 1e9

        yolo_fresh = (now - self.yolo_msg_time) < 1.0

        # primero procesa señales y semáforos si yolo está actualizado
        if yolo_fresh:
            red_detected = False
            yellow_detected = False
            green_detected = False
            tl_seen_this_frame = False

            for det in self.latest_detections:
                area = abs(det.bottom - det.top) * abs(det.right - det.left)
                class_name = det.class_name.strip()

                self.get_logger().info(
                    f'Procesando {class_name} area={area}'
                )

                is_traffic_light = class_name in ['Red', 'Yellow', 'Green']
                min_area = self.traffic_light_min_area if is_traffic_light else self.min_detection_area

                if area < min_area:
                    continue

                if is_traffic_light:
                    tl_seen_this_frame = True
                    self.last_traffic_light_time = now

                    if class_name == 'Red':
                        red_detected = True
                        self.get_logger().warn(
                            f'RED DETECTED area={area}'
                        )
                    elif class_name == 'Yellow':
                        yellow_detected = True
                        self.get_logger().warn(
                            f'YELLOW DETECTED area={area}'
                        )
                    elif class_name == 'Green':
                        green_detected = True
                        self.get_logger().warn(
                            f'GREEN DETECTED area={area}'
                        )

                else:
                    if class_name in self.last_triggered:
                        if (now - self.last_triggered[class_name]) < self.sign_cooldown:
                            continue

                    if class_name == 'Stop':
                        self.last_triggered[class_name] = now
                        self.stop_timer_end = now + 10.0
                        self.get_logger().warn(
                            f'STOP ACTIVATED area={area}'
                        )

                    elif class_name == 'Construction':
                        self.last_triggered[class_name] = now
                        self.construction_timer_end = now + 10.0
                        self.get_logger().warn(
                            f'CONSTRUCTION ACTIVATED area={area}'
                        )

                    elif class_name == 'Give Way':
                        self.last_triggered[class_name] = now
                        self.give_way_timer_end = now + 5.0
                        self.get_logger().warn(
                            f'GIVE WAY ACTIVATED area={area}'
                        )

                    elif class_name == 'Right':
                        if self.executing_maneuver is None and self.pending_maneuver is None:
                            self.last_triggered[class_name] = now
                            self.pending_maneuver = 'RIGHT'
                            self.get_logger().warn(
                                f'RIGHT STORED area={area}'
                            )

                    elif class_name == 'Left':
                        if self.executing_maneuver is None and self.pending_maneuver is None:
                            self.last_triggered[class_name] = now
                            self.pending_maneuver = 'LEFT'
                            self.get_logger().warn(
                                f'LEFT STORED area={area}'
                            )

                    elif class_name == 'Straight':
                        if self.executing_maneuver is None and self.pending_maneuver is None:
                            self.last_triggered[class_name] = now
                            self.pending_maneuver = 'STRAIGHT'
                            self.get_logger().warn(
                                f'STRAIGHT STORED area={area}'
                            )

            # decide qué color de semáforo manda en este frame
            if red_detected:
                self.traffic_light_state = 'RED'
                self.waiting_for_green = True
            elif yellow_detected:
                self.traffic_light_state = 'YELLOW'
            elif green_detected:
                self.traffic_light_state = 'GREEN'
                self.waiting_for_green = False
            elif not tl_seen_this_frame:
                if (now - self.last_traffic_light_time) > self.traffic_light_persist_time:
                    self.traffic_light_state = None

        # prepara la imagen para detectar la línea
        blurred = cv2.GaussianBlur(
            img,
            (5, 5),
            0
        )

        hsv = cv2.cvtColor(
            blurred,
            cv2.COLOR_BGR2HSV
        )

        roi_ratio = self.get_float_param('roi_start_ratio')
        roi_ratio = max(0.0, min(0.95, roi_ratio))

        roi_y_start = int(height * roi_ratio)

        hsv_line_roi = hsv[
            roi_y_start:height,
            0:width
        ]

        found_line, mask_line, cx, cy, line_area, line_cnt = self.detect_line_center(
            hsv_line_roi
        )

        # comando que se va a publicar al final del ciclo
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        error = 0.0
        follow_line = False

        stop_active = self.waiting_for_green or now < self.stop_timer_end

        # prioridad máxima: detenerse si hay rojo o stop activo
        if stop_active:
            if self.waiting_for_green:
                if self.executing_maneuver is not None and not self.maneuver_paused:
                    self.maneuver_paused = True
                    self.maneuver_pause_start_time = now
                    self.get_logger().warn(
                        f'MANEUVER PAUSED due to RED: {self.executing_maneuver} {self.maneuver_step}'
                    )
            else:
                self.executing_maneuver = None
                self.pending_maneuver = None
                self.maneuver_step = None
                self.maneuver_paused = False

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # si una maniobra estaba pausada por rojo, la reanuda
        elif self.maneuver_paused:
            if self.executing_maneuver is not None:
                pause_duration = now - self.maneuver_pause_start_time
                self.maneuver_step_start_time += pause_duration
                self.get_logger().warn(
                    f'MANEUVER RESUMED: {self.executing_maneuver} {self.maneuver_step} '
                    f'(paused for {pause_duration:.1f}s)'
                )

            self.maneuver_paused = False

            if self.executing_maneuver is not None:
                self.run_timed_maneuver(cmd, now)
            elif self.pending_maneuver is not None:
                self.start_timed_maneuver(self.pending_maneuver)
                self.run_timed_maneuver(cmd, now)

        # si ya hay una maniobra ejecutándose, sigue con ella
        elif self.executing_maneuver is not None:
            self.run_timed_maneuver(cmd, now)

        # si hay una maniobra pendiente, la empieza
        elif self.pending_maneuver is not None:
            self.start_timed_maneuver(self.pending_maneuver)
            self.run_timed_maneuver(cmd, now)

        # si no hay maniobra, entonces sigue la línea
        elif found_line:
            follow_line = True

        # si no ve línea, decide si avanza lento o se queda buscando
        else:
            if self.traffic_light_state == 'YELLOW':
                cmd.linear.x = self.v_normal * 0.5
                cmd.angular.z = self.search_angular_speed

            elif now < self.construction_timer_end or now < self.give_way_timer_end:
                cmd.linear.x = self.v_normal * 0.5
                cmd.angular.z = self.search_angular_speed

            else:
                cmd.linear.x = 0.0
                cmd.angular.z = self.search_angular_speed

        # control proporcional para centrar la línea en la imagen
        if follow_line:
            image_center = width / 2.0

            error = (cx - image_center) / image_center

            angular_z = -self.angular_gain * error

            angular_z = max(
                -self.max_angular_speed,
                min(self.max_angular_speed, angular_z)
            )

            if self.traffic_light_state == 'YELLOW':
                cmd.linear.x = self.v_normal * 0.5

            elif now < self.construction_timer_end or now < self.give_way_timer_end:
                cmd.linear.x = self.v_normal * 0.5

            else:
                cmd.linear.x = self.v_normal

            cmd.angular.z = angular_z

        # dibuja la línea detectada y el error visual sobre la imagen
        if found_line and line_cnt is not None:
            real_cy = cy + roi_y_start

            cv2.drawContours(
                img[roi_y_start:height, 0:width],
                [line_cnt],
                -1,
                (0, 255, 0),
                2
            )

            cv2.circle(
                img,
                (cx, real_cy),
                6,
                (0, 0, 255),
                -1
            )

            cv2.line(
                img,
                (int(width / 2.0), roi_y_start),
                (int(width / 2.0), height),
                (255, 0, 0),
                2
            )

            cv2.line(
                img,
                (cx, real_cy),
                (int(width / 2.0), real_cy),
                (0, 255, 255),
                2
            )

        # define el texto de estado que se verá en la imagen procesada
        if self.waiting_for_green or now < self.stop_timer_end:
            state_color = (0, 0, 255)
            state_text = 'STOP'

        elif self.traffic_light_state == 'YELLOW' or now < self.construction_timer_end or now < self.give_way_timer_end:
            state_color = (0, 255, 255)
            state_text = 'SLOW'

        else:
            state_color = (0, 255, 0)
            state_text = 'GO'

        cv2.putText(
            img,
            'STATE: {}'.format(state_text),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            state_color,
            2
        )

        cv2.putText(
            img,
            'Error: {:.2f}'.format(error),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            img,
            'v: {:.2f}  w: {:.2f}'.format(
                cmd.linear.x,
                cmd.angular.z
            ),
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # arma textos extra para saber qué señales están activas
        yolo_texts = []

        if self.traffic_light_state:
            yolo_texts.append(
                'TL: {}'.format(self.traffic_light_state)
            )

        if self.waiting_for_green:
            yolo_texts.append(
                'WAITING FOR GREEN'
            )

        if now < self.stop_timer_end:
            yolo_texts.append(
                'STOP: {:.1f}s'.format(self.stop_timer_end - now)
            )

        if now < self.construction_timer_end:
            yolo_texts.append(
                'CONST: {:.1f}s'.format(self.construction_timer_end - now)
            )

        if now < self.give_way_timer_end:
            yolo_texts.append(
                'GIVE: {:.1f}s'.format(self.give_way_timer_end - now)
            )

        if self.pending_maneuver:
            yolo_texts.append(
                'PEND: {}'.format(self.pending_maneuver)
            )

        if self.executing_maneuver:
            yolo_texts.append(
                'EXEC: {} {}'.format(
                    self.executing_maneuver,
                    self.maneuver_step
                )
            )

        if self.maneuver_paused:
            yolo_texts.append(
                'PAUSED'
            )

        for i, text in enumerate(yolo_texts):
            cv2.putText(
                img,
                text,
                (10, 120 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # manda el comando de movimiento al robot
        self.pub_cmd_vel.publish(cmd)

        # publica la imagen ya dibujada para monitoreo
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(
                img,
                encoding='bgr8'
            )

            self.pub_processed.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(
                'Error publishing processed image: {}'.format(e)
            )


# ============================================================
# arranque del nodo
# ============================================================

def main(args=None):
    rclpy.init(args=args)

    node = TrafficLine()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        stop_msg = Twist()
        node.pub_cmd_vel.publish(stop_msg)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
