"""Figure generation for the pendulum experiments (outputs to docs/images/node/)."""
from .phase import plot_phase_portrait
from .fields import plot_vector_field
from .timeseries import plot_timeseries_seen_unseen
from .growth import plot_error_growth
from .sensitivity import plot_ic_sensitivity
from .figures import save_all_figures

__all__ = ['plot_phase_portrait', 'plot_vector_field', 'plot_timeseries_seen_unseen',
           'plot_error_growth', 'plot_ic_sensitivity', 'save_all_figures']
