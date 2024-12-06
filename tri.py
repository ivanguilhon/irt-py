import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import norm
from scipy.integrate import quad



def logistic_model(theta, alpha, beta, r):
    """
    Logistic model with three parameters.
    Args:
        theta (float): Ability parameter.
        alpha (float): Discrimination parameter.
        beta (float): Difficulty parameter.
        r (float): Guessing parameter.
    Returns:
        float: Probability of success.
    """
    return r + (1.0 - r) / (1 + np.exp(-alpha * (theta - beta)))


def error_probability(theta, alpha=1, beta=5, r=0.2):
    """
    Computes the probability of error.
    Args:
        theta (float): Ability parameter.
        alpha (float): Discrimination parameter.
        beta (float): Difficulty parameter.
        r (float): Guessing parameter.
    Returns:
        float: Probability of error.
    """
    return 1 - logistic_model(theta, alpha, beta, r)


def logistic_no_guessing(theta, alpha=1, beta=5):
    """
    Logistic model without guessing parameter.
    Args:
        theta (float): Ability parameter.
        alpha (float): Discrimination parameter.
        beta (float): Difficulty parameter.
    Returns:
        float: Probability of success.
    """
    return 1.0 / (1 + np.exp(-alpha * (theta - beta)))


def error_probability_no_guessing(theta, alpha=1, beta=5):
    """
    Computes the probability of error without guessing parameter.
    Args:
        theta (float): Ability parameter.
        alpha (float): Discrimination parameter.
        beta (float): Difficulty parameter.
    Returns:
        float: Probability of error.
    """
    return 1 - logistic_no_guessing(theta, alpha, beta)


def weight_function(theta, alpha=1, beta=5, r=0.2):
    """
    Auxiliary weight function for estimation.
    Args:
        theta (float): Ability parameter.
        alpha (float): Discrimination parameter.
        beta (float): Difficulty parameter.
        r (float): Guessing parameter.
    Returns:
        float: Weight value.
    """
    p = logistic_model(theta, alpha, beta, r)
    q = error_probability(theta, alpha, beta, r)
    return logistic_no_guessing(theta, alpha, beta) * error_probability_no_guessing(theta, alpha, beta) / (p * q)


def fisher_information(theta, alpha=0.1, beta=50,r=0.2 ):
    """
    Computes the Fisher information.
    Args:
        theta (float): Ability parameter.
        r (float): Guessing parameter.
        alpha (float): Discrimination parameter.
        beta (float): Difficulty parameter.
    Returns:
        float: Fisher information.
    """
    p = logistic_model(theta, alpha, beta, r)
    return alpha**2 * (1 - p) * ((p - r) / (1 - r))**2 / p


class Question:
    """
    Represents a question in a test with parameters and response analysis.
    """
    def __init__(self, question_index=0, alpha=1.0, beta=0, guessing_param=0.2):
        """
        Initialize the Question instance.
        Args:
            question_index (int): Index of the question.
            responses (DataFrame): Responses data.
        """
        self.question_index = question_index
        self.correct_answer = 'Z'
        self.alpha = alpha
        self.beta = beta
        self.guessing_param = guessing_param

    def estimate_parameters(self, responses, num_steps=100):
        """
        Estimate the parameters of the question using curve fitting.
        Args:
            responses (DataFrame): Responses data.
            num_steps (int): Number of steps for binning the data.
        Returns:
            ndarray: Estimated parameters [alpha, beta, guessing].
        """
        step_size = int(self.num_responses / num_steps)
        k = 0
        mean_points = responses['Standardized Points'].mean()
        std_points = responses['Standardized Points'].std()
        ability_summary = []
        correct_summary = []
        sorted_responses = responses.sort_values('Standardized Points', ascending=True)
        abilities = (sorted_responses['Standardized Points'] - mean_points) / std_points
        correct_pattern = sorted_responses[self.question_index]

        while (k + 1) * step_size < self.num_responses:
            ability_summary.append(abilities[k * step_size:(k + 1) * step_size].mean())
            correct_summary.append(correct_pattern[k * step_size:(k + 1) * step_size].mean())
            k += 1

        popt, _ = opt.curve_fit(
            logistic_model, 
            np.array(ability_summary), 
            np.array(correct_summary), 
            p0=[0.1, 3, 0.2], 
            bounds=([0, -5, 0], [1.5, 5, 1])
        )

        self.expected_correct_pattern = [np.asarray(ability_summary), np.asarray(correct_summary)]
        self.set_alpha(popt[0])
        self.set_beta(popt[1])
        self.set_guessing_param(popt[2])
        return popt

    def calculate_correct_responses(self, question_index):
        """
        Extracts the responses for a specific question.
        Args:
            question_index (int): Index of the question.
        Returns:
            Series: Correct responses for the question.
        """
        return self.responses

    def prior_probability_correct(self):
        """
        Calculate a prior probability of correctness.
        """
        f =lambda x: ( norm.pdf(x, loc=0.0, scale=1.0) * self.probability_correct(x) )
        


        return quad(f,-4,4)[0]

    def set_alpha(self, alpha):
        """
        Set the discrimination parameter alpha.
        Args:
            alpha (float): New value for alpha.
        """
        self.alpha = alpha

    def set_beta(self, beta):
        """
        Set the difficulty parameter beta.
        Args:
            beta (float): New value for beta.
        Returns:
            float: Updated beta value.
        """
        self.beta = beta
        return beta

    def set_guessing_param(self, guessing_param):
        """
        Set the guessing parameter.
        Args:
            guessing_param (float): New value for the guessing parameter.
        Returns:
            float: Updated guessing parameter value.
        """
        self.guessing_param = guessing_param
        return guessing_param

    # Probability functions
    def probability_correct(self, theta):
        """
        Compute the probability of a correct response based on the 3PL model.
        Args:
            theta (float): Ability parameter.
        Returns:
            float: Probability of a correct response.
        """
        return self.guessing_param + (1.0 - self.guessing_param) / (1 + np.exp(-self.alpha * (theta - self.beta)))

    def probability_incorrect(self, theta):
        """
        Compute the probability of an incorrect response.
        Args:
            theta (float): Ability parameter.
        Returns:
            float: Probability of an incorrect response.
        """
        return 1 - self.probability_correct(theta)

    def probability_correct_no_guessing(self, theta):
        """
        Compute the probability of a correct response without the guessing parameter.
        Args:
            theta (float): Ability parameter.
        Returns:
            float: Probability of a correct response without guessing.
        """
        return 1.0 / (1 + np.exp(-self.alpha * (theta - self.beta)))

    def probability_incorrect_no_guessing(self, theta):
        """
        Compute the probability of an incorrect response without the guessing parameter.
        Args:
            theta (float): Ability parameter.
        Returns:
            float: Probability of an incorrect response without guessing.
        """
        return 1 - self.probability_correct_no_guessing(theta)

    def weight_function(self, theta):
        """
        Auxiliary weight function for parameter estimation.
        Args:
            theta (float): Ability parameter.
        Returns:
            float: Weight value.
        """
        p = self.probability_correct(theta)
        q = self.probability_incorrect(theta)
        return (
            self.probability_correct_no_guessing(theta)
            * self.probability_incorrect_no_guessing(theta)
            / (p * q)
        )

    def fisher_information(self, theta, guessing_param=0.2, alpha=0.1, beta=50):
        """
        Compute the Fisher information for the question.
        Args:
            theta (float): Ability parameter.
            guessing_param (float): Guessing parameter.
            alpha (float): Discrimination parameter.
            beta (float): Difficulty parameter.
        Returns:
            float: Fisher information.
        """
        p = self.probability_correct(theta)
        return alpha**2 * (1 - p) * ((p - guessing_param) / (1 - guessing_param))**2 / p

    def make_figure(
        self,
        target_folder='./',
        file_name='question',
        file_format='.eps',
        reference_curve=True,
        title='Question',
        mean_ability=0,
        std_ability=1,
        show_experimental=True
    ):
        """
        Generates and saves a plot for the question's probability curve and Fisher information.
        
        Args:
            target_folder (str): Folder path to save the figure.
            file_name (str): Name of the output file.
            file_format (str): File format for the saved figure (e.g., '.eps', '.png').
            reference_curve (bool): Whether to plot a reference curve for comparison.
            title (str): Title of the question curve.
            mean_ability (float): Mean ability level for rescaling.
            std_ability (float): Standard deviation of ability for rescaling.
            show_experimental (bool): Whether to display experimental data points.
        """
        x_values = np.linspace(0, 100, 100)

        # Plot experimental data points
        if show_experimental:
            plt.plot(
                self.expected_correct_pattern[0] * std_ability + mean_ability,
                self.expected_correct_pattern[1],
                linewidth=0,
                marker='o',
                alpha=0.1,
                color='blue',
                markersize=1.5
            )

        # Plot reference curve
        if reference_curve:
            plt.plot(
                x_values,
                logistic_model(x_values, r=0.15, alpha=0.1, beta=65),
                label='Reference',
                color='red'
            )
            plt.ylim(0, 1)
            plt.plot(
                x_values,
                100 * fisher_information(x_values, r=0.15, alpha=0.1, beta=65),
                linestyle='dashed',
                color='red'
            )

        # Plot question-specific curve
        plt.plot(
            x_values,
            self.probability_correct((x_values - mean_ability) / std_ability),
            label=title
        )
        plt.plot(
            x_values,
            self.fisher_information((x_values - mean_ability) / std_ability),
            linestyle='dashed',
            color=plt.gca().lines[-1].get_color()
        )

        # Add legend and grid
        plt.legend()
        plt.grid()

        # Save and clear the plot
        plt.savefig(f"{target_folder}{file_name}{file_format}")
        plt.cla()


class Student:
    """
    Represents a student in the test analysis.
    """

    def __init__(self, student_index, responses, parameters):
        """
        Initialize the Student instance.
        Args:
            student_index (int): Index of the student.
            responses (DataFrame): Responses data.
            parameters (DataFrame): Parameters for the questions.
        """
        self.index = student_index
        self.data = responses[responses['candidate'] == student_index]
        self.parameters = parameters
        self.correct_responses = self.get_correct_responses()
        self.ability = float(self.data['ability'])

    def get_correct_responses(self):
        """
        Retrieve the student's responses for all questions.
        Returns:
            ndarray: Array of responses (0 for incorrect, 1 for correct).
        """
        num_questions = 60  # Number of questions
        responses = []
        for k in range(1, num_questions + 1):
            responses.append(self.data[k])
        return np.asarray(responses)

    def log_likelihood(self, theta):
        """
        Compute the log-likelihood for a given ability parameter.
        Args:
            theta (float): Ability parameter.
        Returns:
            float: Log-likelihood value.
        """
        self.parameters['null'] = (
            (self.parameters['prob_correct'] > 0.999)
            | ((self.parameters['question'] >= 25) & (self.parameters['question'] <= 36))
        )
        self.parameters['null'] = self.parameters['null'].replace({True: 1.0, False: 0.0})

        self.parameters['P'] = self.parameters['guessing'] + (
            1.0 - self.parameters['guessing']
        ) / (1 + np.exp(-self.parameters['discrimination'] * (theta - self.parameters['difficulty'])))
        self.parameters['Q'] = 1 - self.parameters['P']
        self.parameters['response_ij'] = self.correct_responses
        self.parameters['log_likelihood_ij'] = (
            1 - self.parameters['null']
        ) * (
            self.parameters['response_ij'] * np.log(self.parameters['P'])
            + (1 - self.parameters['response_ij']) * np.log(self.parameters['Q'])
        )
        return self.parameters['log_likelihood_ij'].sum()

    def negative_log_likelihood(self, theta):
        """
        Compute the negative log-likelihood for optimization.
        Args:
            theta (float): Ability parameter.
        Returns:
            float: Negative log-likelihood value.
        """
        return -self.log_likelihood(theta)

    def set_ability(self, theta):
        """
        Set the student's ability parameter.
        Args:
            theta (float): New ability value.
        Returns:
            float: Updated ability value.
        """
        self.ability = float(theta)
        return self.ability

    def calculate_ability(self, method='NM'):
        """
        Calculate the student's ability using optimization.
        Args:
            method (str): Optimization method ('NM' for Nelder-Mead, 'NR' for Newton-Raphson).
        Returns:
            float: Estimated ability.
        """
        theta_initial = self.ability

        if method == 'NM':  # Nelder-Mead method
            res = opt.minimize(self.negative_log_likelihood, theta_initial, method='Nelder-Mead', tol=1e-4)
            theta = res.x[0]
            self.set_ability(theta)
            return theta

        if method == 'NR':  # Newton-Raphson method
            # Placeholder for Newton-Raphson logic
            pass

    def print_summary(self):
        """
        Print a summary of the student's details.
        Returns:
            bool: Always True.
        """
        print('Index:', self.index)
        print('Ability:', self.ability)
        print('Log(L):', self.log_likelihood(self.ability))
        return True
