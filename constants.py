
etype_map = {
    'clicks': 0,
    'carts': 1,
    'orders': 2
}

etype_map_reverse = {v: k for k, v in etype_map.items()}

time_encoding_max_minutes = 128
time_encoding_max_hours = 64
time_encoding_max_days = 64