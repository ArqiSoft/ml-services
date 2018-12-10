"""
Module which include all needed methods for render html templates
"""

import os

from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa

PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False, loader=FileSystemLoader(PATH), trim_blocks=False)
# define templates names for usage
# DO NOT use template names as strings, use this variables
TRAINING_TEMPLATE = 'training_report.html'
QMRF_TEMPLATE = 'QMRF_report.html'


def create_report_html(context, path_to_template, html_report_path):
    """
    Method which render template with context and save it to html file
    Return path to saved html file

    :param context: context for rendering
    :param path_to_template: path to template file
    :param html_report_path: path to write rendered html report file
    :type path_to_template: str
    :type html_report_path: str
    :return: path to rendered html report file
    :rtype: str
    """

    # render template with context
    html = render_template('templates/{}'.format(path_to_template), context)

    # save rendered template to html file
    html_report_file = open(html_report_path, 'w')
    html_report_file.write(html)
    html_report_file.close()

    return html_report_path


def render_template(path_to_template, context):
    """
    Method for render any template with any context (if possible)

    :param path_to_template: path to template file
    :param context: context for rendering
    :type path_to_template: str
    :return: rendered template
    """

    return TEMPLATE_ENVIRONMENT.get_template(path_to_template).render(context)


def make_pdf_report(save_folder, context, model_name=None):
    """
    Method which make model training report as pdf and save it to file

    :param context:
    :param model_name: model name which user define
    :type model_name: str
    :return: path to training report pdf file
    :rtype: str
    """

    # make report as html page
    html_report_path = os.path.join(save_folder, TRAINING_TEMPLATE)
    create_report_html(
        context, TRAINING_TEMPLATE, html_report_path)
    # convert report to pdf and save to file
    pdf_report_path = os.path.join(
        save_folder, '{}_report.pdf'.format(model_name))
    pisa.CreatePDF(
        open(html_report_path, 'r'), open(pdf_report_path, 'wb'),
        path=os.path.abspath(save_folder)
    )
    pdf_path = os.path.abspath(pdf_report_path)

    return pdf_path
