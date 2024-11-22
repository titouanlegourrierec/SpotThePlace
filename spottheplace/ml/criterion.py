import torch
import torch.nn as nn


class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Convert degrees to radians
        y_pred_rad = torch.deg2rad(y_pred)
        y_true_rad = torch.deg2rad(y_true)

        # Extract latitude and longitude
        lat1, lon1 = y_pred_rad[:, 1], y_pred_rad[:, 0]
        lat2, lon2 = y_true_rad[:, 1], y_true_rad[:, 0]

        # Compute the difference between the two points
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        # Compute the distance between the two points using the Haversine formula
        a = torch.sin(delta_lat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        # Radius of the Earth in kilometers
        R = 6371

        # Return the geodesic distance between the two points in kilometers
        return c.mean() * R
