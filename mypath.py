class Path(object):
	def __init__(self):
		self.event = '/home/tan/Workspace/Yuki_challenge/snow_detect_simple'
					# path_to_event_folder/snow_detect

		self.patch = '/home/tan/Workspace/Dataset/Yuki_challenge/harmo_patch'
					# 'path_to_patches_folder/hormo_patch'

		self.harmo = '/home/tan/Workspace/Dataset/Yuki_challenge/harmo_lab'
					# 'path_to_original_dataset/harmo_harmo-ml-challenge-2019-0922'
		
		self.detect = '/home/tan/Workspace/Yuki_challenge/snow_detect_simple/detect_list.txt'
			# return 'path of the text file is received as an argument for program execution'