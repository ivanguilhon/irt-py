import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import norm
from scipy.integrate import quad


def logistic_model(theta, alpha, beta, c):
    """
    Logistic model with three parameters.
    Args:
        theta (float): Ability parameter.
        alpha (float): Discrimination parameter.
        beta (float): Difficulty parameter.
        c(float): Guessing parameter.
    Returns:
        float: Probability of success.
    """
    return c + (1.0 - c) / (1 + np.exp(-alpha * (theta - beta)))


def error_probability(theta, alpha=1, beta=5, c=0.2):
    """
    Computes the probability of error.
    Args:
        theta (float): Ability parameter.
        alpha (float): Discrimination parameter.
        beta (float): Difficulty parameter.
        c (float): Guessing parameter.
    Returns:
        float: Probability of error.
    """
    return 1 - logistic_model(theta, alpha, beta, c)


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


def weight_function(theta, alpha=1, beta=5, c=0.2):
    """
    Auxiliary weight function for estimation.
    Args:
        theta (float): Ability parameter.
        alpha (float): Discrimination parameter.
        beta (float): Difficulty parameter.
        c (float): Guessing parameter.
    Returns:
        float: Weight value.
    """
    p = logistic_model(theta, alpha, beta, c)
    q = error_probability(theta, alpha, beta, c)
    return (
        logistic_no_guessing(theta, alpha, beta)
        * error_probability_no_guessing(theta, alpha, beta)
        / (p * q)
    )


def fisher_information(theta, alpha=0.1, beta=50, r=0.2):
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
    return alpha**2 * (1 - p) * ((p - r) / (1 - r)) ** 2 / p


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
        self.correct_answer = "Z"
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
        mean_points = responses["Standardized Points"].mean()
        std_points = responses["Standardized Points"].std()
        ability_summary = []
        correct_summary = []
        sorted_responses = responses.sort_values("Standardized Points", ascending=True)
        abilities = (sorted_responses["Standardized Points"] - mean_points) / std_points
        correct_pattern = sorted_responses[self.question_index]

        while (k + 1) * step_size < self.num_responses:
            ability_summary.append(
                abilities[k * step_size : (k + 1) * step_size].mean()
            )
            correct_summary.append(
                correct_pattern[k * step_size : (k + 1) * step_size].mean()
            )
            k += 1

        popt, _ = opt.curve_fit(
            logistic_model,
            np.array(ability_summary),
            np.array(correct_summary),
            p0=[0.1, 3, 0.2],
            bounds=([0, -5, 0], [1.5, 5, 1]),
        )

        self.expected_correct_pattern = [
            np.asarray(ability_summary),
            np.asarray(correct_summary),
        ]
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
        f = lambda x: (norm.pdf(x, loc=0.0, scale=1.0) * self.probability_correct(x))

        return quad(f, -4, 4)[0]

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
        return self.guessing_param + (1.0 - self.guessing_param) / (
            1 + np.exp(-self.alpha * (theta - self.beta))
        )

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

    def fisher_information(self, theta):
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
        return (
            self.alpha**2
            * (1 - p)
            * ((p - self.guessing_param) / (1 - self.guessing_param)) ** 2
            / p
        )

    def make_figure(
        self,
        target_folder="./",
        file_name="question",
        file_format=".eps",
        reference_curve=True,
        title="Question",
        xlim=[-4, 4],
        mean_ability=0,
        std_ability=1,
        show_experimental=True,
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
        x_values = np.linspace(xlim[0], xlim[1], 100)

        # Plot experimental data points
        if show_experimental:
            plt.plot(linewidth=0, marker="o", alpha=0.1, color="blue", markersize=1.5)

        # Plot reference curve
        if reference_curve:
            plt.plot(
                x_values,
                logistic_model(x_values, r=0.15, alpha=2, beta=0),
                label="Reference",
                color="red",
            )
            plt.ylim(0, 1)
            plt.plot(
                x_values,
                fisher_information(x_values, r=0.15, alpha=2, beta=0),
                linestyle="dashed",
                color="red",
            )
        plt.xlim(xlim[0], xlim[1])

        # Plot question-specific curve
        plt.plot(
            x_values,
            self.probability_correct((x_values - mean_ability) / std_ability),
            label=title,
        )
        plt.plot(
            x_values,
            self.fisher_information((x_values - mean_ability) / std_ability),
            linestyle="dashed",
            color=plt.gca().lines[-1].get_color(),
        )

        # Add legend and grid
        plt.legend()
        plt.grid()

        # Save and clear the plot
        plt.savefig(f"{target_folder}{file_name}{file_format}")
        plt.cla()


def P_3PL(theta, a_i, b_i, c_i):
    return c_i + (1 - c_i) / (1 + np.exp(-a_i * (theta - b_i)))


class Student:
    """
    Represents a student in the test analysis.
    """

    def __init__(self, student_index=0, responses=np.nan, ability=0, parameters=None):
        """
        Initialize the Student instance.
        Args:
            student_index (int): Index of the student.
            responses (ndarray): Array of responses (0 for incorrect, 1 for correct).
            ability (Float): ability of the student, to be given or estimated by irt.

        """
        self.index = student_index
        self.responses = responses
        self.ability = ability
        self.parameters = parameters

    def set_parameters(self, parameters):
        self.parameters = parameters
        return True

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

    def log_likelihood(self, theta: float = np.nan) -> float:
        """
        Compute the log-likelihood for a given ability parameter.
        Args:
            theta (float): Ability parameter.
        Returns:
            float: Log-likelihood value.
        """
        if np.isnan(theta):
            theta = self.ability

        self.parameters["P"] = self.parameters["c"] + (1.0 - self.parameters["c"]) / (
            1 + np.exp(-self.parameters["a"] * (theta - self.parameters["b"]))
        )
        self.parameters["Q"] = 1 - self.parameters["P"]
        self.parameters["response_ij"] = self.responses
        self.parameters["log_likelihood_ij"] = (1 - self.parameters["null"]) * (
            self.parameters["response_ij"] * np.log(self.parameters["P"])
            + (1 - self.parameters["response_ij"]) * np.log(self.parameters["Q"])
        )
        return self.parameters["log_likelihood_ij"].sum()

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

    def likelihood(self, theta=np.nan):
        if np.isnan(theta):
            theta = self.ability

        prob = [
            logistic_model(
                theta,
                self.parameters["a"][i],
                self.parameters["b"][i],
                self.parameters["c"][i],
            )
            for i in range(len(self.responses))
        ]
        return np.prod([p if r == 1 else (1 - p) for p, r in zip(prob, self.responses)])

    def calculate_ability(self, method="EAP", loc_prior=0, scale_prior=1):
        """
        Calculate the student's ability using optimization.
        Args:
            method (str): Optimization method ('NM' for Nelder-Mead, 'NR' for Newton-Raphson).
        Returns:
            float: Estimated ability.
        """
        theta_initial = self.ability

        if method == "NM":  # Nelder-Mead method
            res = opt.minimize(
                self.negative_log_likelihood,
                theta_initial,
                method="Nelder-Mead",
                tol=1e-4,
            )
            theta = res.x[0]
            self.set_ability(theta)
            return theta

        if method == "NR":  # Newton-Raphson method
            # Placeholder for Newton-Raphson logic
            pass

        if method == "EAP":  # Expected a posteriori
            # Adjusted a posteriori probability distribution
            def posterior(theta, a, b, c, responses):
                prior = norm.pdf(
                    theta, loc=loc_prior, scale=scale_prior
                )  # Prior normal distribution
                return self.likelihood(theta) * prior

            # Estimação por EAP ajustada
            def eap(a, b, c, responses):
                numerator = lambda theta: theta * posterior(
                    theta,
                    self.parameters["a"],
                    self.parameters["b"],
                    self.parameters["c"],
                    self.responses,
                )
                denominator = lambda theta: posterior(
                    theta,
                    self.parameters["a"],
                    self.parameters["b"],
                    self.parameters["c"],
                    self.responses,
                )

                num = quad(numerator, -10, 10)[0]
                denom = quad(denominator, -10, 10)[0]
                return num / denom

            self.ability = eap(
                self.parameters["a"],
                self.parameters["b"],
                self.parameters["c"],
                self.responses,
            )
            return self.ability

    def print_summary(self):
        """
        Print a summary of the student's details.
        Returns:
            bool: Always True.
        """
        print("Index:", self.index)
        print("Ability:", self.ability)
        # print('Log(L):', self.log_likelihood(self.ability))
        return True


#######################################  TESTING SECTION ######################################33

###  Testing question class

# print('Testing Question Class...')
# question=Question()
# print('Default parameters:',question.alpha, question.beta, question.guessing_param)
# question.set_alpha(1.5)
# question.set_beta(-0.2)
# question.set_guessing_param(0.15)
# print('Default parameters:',question.alpha, question.beta, question.guessing_param)
# print('P(X=1|theta=0):',question.probability_correct(0))
# print('Correctness a priori probability:',question.prior_probability_correct())
# question.make_figure()

###Fictious test
# a = np.asarray([1.0, 1.0, 1])  # discriminação
# b = np.asarray([1.0, 3.0,3] ) # dificuldade
# c = np.asarray([0.2, 0.2,0.2])  # acerto ao acaso (para 3PL)
# null=np.asarray([0,0,0])
# parametersStudents={'a':a,'b':b,'c':c,'null':null}


### Testing student class
# responsesStudent = [1, 0,1]  # padrão de respostas (1 = correto, 0 = incorreto)
# student=Student(ability=2,responses=responsesStudent,parameters=parametersStudents)
# student.print_summary()
# student.calculate_ability(method='EAP')
# student.print_summary()
