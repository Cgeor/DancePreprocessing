import json
'''Generate json file containing the camera intrinsics in a format readable by multiviewcalib.'''
sensor_size_x = 10
sensor_size_y = 5.625
dim_x = 1920
dim_y = 1080

c_x = dim_x / sensor_size_x
c_y = dim_y / sensor_size_y
scale = 10

# 4.2
f0 = 0.548192918 * scale
K0 = [[f0 * c_x,  0.000000000,  0.480572104 * dim_x]
       ,[0.000000000,  f0 * c_y,  0.280282825 * dim_x]
       ,[0.000000000,  0.000000000,  1.000000000]]
dist0 = [[-0.105430894,	  0.162188932,	  0.003710969,	  0.000701237,	 -0.087060384]]
# 4.0
f1 = 0.548929513 * scale
K1 = [[f1 * c_x,  0.000000000,  0.494857967 * dim_x]
       ,[0.000000000,  f1 * c_y,  0.280436277 * dim_x]
       ,[0.000000000,  0.000000000,  1.000000000]]
dist1 = [[-0.106410407,	  0.166353703,	  0.002218384,	  0.005659800,	 -0.095405526]]
# 6.0
f2 = 0.536758661 * scale
K2 = [[f2 * c_x,  0.000000000,  0.487233520 * dim_x]
       ,[0.000000000,  f2 * c_y,  0.284947038 * dim_x]
       ,[0.000000000,  0.000000000,  1.000000000]]
dist2 = [[-0.028459698,	  0.048451919,	  0.002094091,	 -0.006526705,	 -0.035562489]]
# 6.1
f3 = 0.537665427 * scale
K3 = [[f3 * c_x,  0.000000000,  0.496364772 * dim_x]
       ,[0.000000000,  f3 * c_y,  0.283008814 * dim_x]
       ,[0.000000000,  0.000000000,  1.000000000]]
dist3 = [[-0.032690108,	  0.067707039,	  0.003742763,	  0.000895648,	 -0.057245590]]
# 6.2
f4 = 0.534982204 * scale
K4 = [[f4 * c_x,  0.000000000,  0.502504468 * dim_x]
       ,[0.000000000,  f4 * c_y,  0.27560851 * dim_x]
       ,[0.000000000,  0.000000000,  1.000000000]]
dist4 = [[-0.014846570,	  0.013632424,	 -0.000825007,	  0.001156945,	 -0.010639322]]
# 6.3
f5 = 0.535578609 * scale
K5 = [[f5 * c_x,  0.000000000,  0.495234072 * dim_x]
       ,[0.000000000,  f5 * c_y,  0.283003330 * dim_x]
       ,[0.000000000,  0.000000000,  1.000000000]]
dist5 = [[-0.023071954,	  0.040432706,	  0.001981513,	 -0.000437010,	 -0.033381123]]
# 6.4
f6 = 0.545515299 * scale
K6 = [[f6 * c_x,  0.000000000,  0.493658721 * dim_x]
       ,[0.000000000,  f6 * c_y,  0.280745953 * dim_x]
       ,[0.000000000,  0.000000000,  1.000000000]]
dist6 = [[-0.017756484,	  0.022002656,	  0.001094042,	  0.002050543,	 -0.019478260]]
# G2
f7 = 0.612473607 * scale
K7 = [[f7 * c_x,  0.000000000,  0.525920093 * dim_x]
       ,[0.000000000,  f7 * c_y,  0.296525657 * dim_x]
       ,[0.000000000,  0.000000000,  1.000000000]]
dist7 = [[-0.079516798,	  0.061645985,	  0.003682427,	  0.013892549,	 -0.045157980]]

def camera_dict(K, dist):
    return {"K": K, "dist": dist}

dict = {}
# for seq 2 - 9
dict["cam0"] = camera_dict(K1, dist1) # 4.0
dict["cam1"] = camera_dict(K0, dist0) # 4.2
dict["cam2"] = camera_dict(K2, dist2) # 6.0
dict["cam3"] = camera_dict(K3, dist3) # 6.1
dict["cam4"] = camera_dict(K4, dist4) # 6.2
dict["cam5"] = camera_dict(K5, dist5) # 6.3
dict["cam6"] = camera_dict(K7, dist7) # G2
#dict["cam7"] = camera_dict(K7, dist7) # 6.4

'''
# for seq 1
dict["cam0"] = camera_dict(K0, dist0)
dict["cam1"] = camera_dict(K2, dist2)
dict["cam2"] = camera_dict(K3, dist3)
dict["cam3"] = camera_dict(K5, dist5)
dict["cam4"] = camera_dict(K7, dist7)
'''

with open('/home/costa/src/multiview_calib/examples/seq7/intrinsics.json', 'w') as outfile:
    json.dump(dict, outfile, indent=2)