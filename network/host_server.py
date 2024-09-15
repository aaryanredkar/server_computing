from flask import Flask, render_template, jsonify, request
import psutil
import socket
from datetime import datetime, timedelta

app = Flask(__name__)

# In-memory store for machine stats
machines_data = {}
last_update_times = {}

# Route to render the main dashboard page
@app.route('/')
def index():
    # Check for machines that haven't sent data in a while and mark as 'Off'
    for hostname, last_update in last_update_times.items():
        if datetime.now() - last_update > timedelta(minutes=2):  # No data in 2 minutes
            machines_data[hostname]['machine_state'] = 'Off'

    return render_template('dashboard.html')

# Route to get detailed info for a specific machine
@app.route('/machine/<hostname>')
def machine_details(hostname):
    machine = machines_data.get(hostname, {})
    return render_template('machine_details.html', machine=machine)

# Route for clients to send data
@app.route('/update_machine', methods=['POST'])
def update_machine():
    data = request.json
    hostname = data['hostname']
    machines_data[hostname] = data
    last_update_times[hostname] = datetime.now()  # Update the timestamp
    return jsonify(success=True)

# Route to provide machine data in JSON format for live updating
@app.route('/live_data')
def live_data():
    return jsonify(machines_data)

# Utility function to get the host machine's IP address
def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == '__main__':
    host_ip = get_ip_address()
    app.run(host=host_ip, port=5000, debug=True)
