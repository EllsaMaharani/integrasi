#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import serial
import time
import random
import csv
import os

class RFIDSensorPublisher:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        # Inisialisasi node ROS
        rospy.init_node('rfid_sensor_publisher', anonymous=True)
        
        # Buat publisher
        self.sensor_pub = rospy.Publisher('/rfid_sensor_data', String, queue_size=10)
        
        # Coba buka koneksi serial dengan ESP32
        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            print(f"📡 ESP32 UART Serial connected on {port}")
            print(f"Publisher sensor: /rfid_sensor_data")
        except Exception as e:
            print(f"⚠️ Serial connection error: {e}")
            print(f"⚠️ Will continue in simulation mode")
        
        # Parameter simulasi
        self.simulation_counter = 1000
        
        # Run loop
        self.run()
    
    def run(self):
        """Loop utama untuk membaca data dan publish"""
        rate = rospy.Rate(10)  # 10 Hz
        print("🚀 RFID Sensor Publisher started - Reading from ESP32")
        
        while not rospy.is_shutdown():
            rfid_data = None
            
            # Coba baca data dari ESP32
            if self.ser:
                try:
                    if self.ser.in_waiting > 0:
                        serial_data = self.ser.readline().decode().strip()
                        print(f"📥 Raw data from ESP32: {serial_data}")
                        
                        # Check if data already contains RSSI
                        if ',' in serial_data:
                            rfid_data = serial_data  # Already in format "tag_id,rssi"
                        else:
                            # Add a simulated RSSI if ESP32 only sends tag ID
                            tag_id = serial_data
                            rssi = random.randint(-95, -60)
                            rfid_data = f"{tag_id},{rssi}"
                            print(f"⚠️ RSSI not in data, added simulated value: {rssi}")
                except Exception as e:
                    print(f"⚠️ Serial read error: {e}")
            
            # Publish data jika tersedia
            if rfid_data:
                self.sensor_pub.publish(rfid_data)
                print(f"📤 Published: {rfid_data}")
            
            rate.sleep()

if __name__ == "__main__":
    try:
        # Cek port serial yang mungkin ada
        possible_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
        
        # Deteksi port yang aktif
        port_to_use = None
        for port in possible_ports:
            try:
                ser = serial.Serial(port, 115200, timeout=0.1)
                ser.close()
                port_to_use = port
                print(f"✅ Found ESP32 on port: {port}")
                break
            except:
                continue
        
        # Gunakan port yang ditemukan atau default
        if port_to_use:
            publisher = RFIDSensorPublisher(port=port_to_use)
        else:
            print("⚠️ No ESP32 detected, using default port")
            publisher = RFIDSensorPublisher()
    except rospy.ROSInterruptException:
        pass