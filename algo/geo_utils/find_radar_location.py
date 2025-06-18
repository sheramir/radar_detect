"""Find Radar location for 3 IQ files (example test).
Created on Mon Jun 22 10:10:17 2020
@author: v025222357 Amir Sher
"""
import os.path as path
from algo.RF_Classes.IQData import IQData
from algo.geo_utils.tdoa import tdoa
from algo.geo_utils.geolocation_solve_tdoa import solve_pos_by_tdoa
from sensors_metadata.sensors import Sensors
from utils.gis.gis import longlat2utm
from utils.gis.plotmap import mymap


all_sensors = Sensors()
all_sensors.calculate_sensors_distance_matrix()
#df

in_path = r'C:\IQ_Data\Radar\sync_sensors\dataset-tri'
out_path = in_path

file_base = '008_{}_Rada_08012020_022227.xdat'
iq_list = []
sensors_pos = []
tdoa_list = []

mysensors = ['Kisufim', 'Nahaloz', 'Yiftah']
for sensor in mysensors:
    long = all_sensors.get_sensor_data(sensor)['longitude']
    lat = all_sensors.get_sensor_data(sensor)['latitude']
    utm = longlat2utm((long, lat))
    sensors_pos.append(utm[0])

    iq_file = path.join(in_path, file_base.format(sensor))
    iq_data = IQData(iq_file, start_time=0.02, sample_length=0.06)
    iq_list.append(iq_data.iq)
ts = iq_data.ts

#iq_data[i].plot_iq(plot1='time.real')
#iq_data[i].plot_iq(plot1='psd')

for s1, sensor1 in enumerate(mysensors[:-1]):
    for s2, sensor2 in enumerate(mysensors[s1+1:], s1+1):
        delay, dist = tdoa(iq_list[s2], iq_list[s1], ts, showplot=False)
        tdoa_list.append(dist)
        print(f'Time delay {sensor1} {sensor2}: {delay}')
        print(f'Distance {sensor1} {sensor2}: {dist:0.3f} meters')


radar_location, radar_polygon = solve_pos_by_tdoa(sensors_pos, tdoa_list)

newmap = mymap()
newmap.get_sensors()
newmap.add_point(*radar_location, name='RADA_Location', color='r', marker='D')
#newmap.add_polygon(radar_location, name='RADA_Location')

newmap.plotmap(map_ID='OTEF GAZA', title='Geo-location')
newmap.plotmap(map_ID='ISRAEL SOUTH')
#newmap.plotmap(map_ID='ISRAEL CENTER')
#newmap.plotmap(map_ID='ISRAEL')
#newmap.plotmap(map_ID='ISRAEL_CENTER NORTH')
newmap.plotmap(map_ID='ISRAEL_NORTH')