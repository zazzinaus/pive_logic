import re
import os
from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
from .fol_prover9_parser import Prover9_FOL_Formula # . all'inizio
from .Formula import FOL_Formula # . all'inizio

# set the path to the prover9 executable
#os.environ['PROVER9'] = '../Prover9/bin'
os.environ['PROVER9'] = './symbolic_solvers/Prover9/bin'


class FOL_Prover9_Program:
    def __init__(self, premises_string: str, conclusion_string: str, dataset_name='FOLIO') -> None:
        self.premises_string = premises_string
        self.conclusion_string = conclusion_string
        self.invalid_premises = []
        self.invalid_conclusion = None
        self.flag = self.parse_logic_program()
        self.dataset_name = dataset_name

    # def parse_logic_program(self):
    #     try:
    #         # Split the string into premises and conclusion
    #         premises = self.premises_string.strip().split('\n')
    #         conclusion = self.conclusion_string.strip().split('\n')

    #         # Store the premises and conclusion in the object
    #         self.logic_premises = [p for p in premises]
    #         self.logic_conclusion = [c for c in conclusion]
    #         print(self.logic_premises)
    #         print(self.logic_conclusion)
    #         # Convert premises to Prover9 format
    #         self.prover9_premises = []
    #         for premise in self.logic_premises:
    #             fol_rule = FOL_Formula(premise)
                
    #             if not fol_rule.is_valid:
    #                 return False
    #             prover9_rule = Prover9_FOL_Formula(fol_rule)
    #             self.prover9_premises.append(prover9_rule.formula)

    #         # Convert conclusion to Prover9 format
    #         fol_conclusion = FOL_Formula(self.logic_conclusion[0])
    #         if not fol_conclusion.is_valid:
    #             return False
    #         self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
    #         return True
    #     except Exception as e:
    #         print(f"Error while parsing logic program: {e}")
    #         return False

    def parse_logic_program(self):
            try:
                # Split the string into premises and conclusion, skipping headers
                premises = self.premises_string.strip().split('\n')
                conclusion = self.conclusion_string.strip().split('\n')

                # Filter out header lines
                premises = [p for p in premises if not p.startswith('###')]
                conclusion = [c for c in conclusion if not c.startswith('###')]

                # Store the premises and conclusion in the object
                self.logic_premises = premises
                self.logic_conclusion = conclusion

                # Convert premises to Prover9 format
                self.prover9_premises = []
                for premise in self.logic_premises:
                    fol_rule = FOL_Formula(premise)
                    if not fol_rule.is_valid:
                        self.invalid_premises.append(premise)
                        continue  # Skip invalid premises
                    prover9_rule = Prover9_FOL_Formula(fol_rule)
                    self.prover9_premises.append(prover9_rule.formula)

                # Convert conclusion to Prover9 format
                if len(self.logic_conclusion) > 0:
                    fol_conclusion = FOL_Formula(self.logic_conclusion[0])
                    if not fol_conclusion.is_valid:
                        self.invalid_conclusion = self.logic_conclusion[0]
                        return False
                    self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
                else:
                    self.invalid_conclusion = "No conclusion provided"
                    return False

                return True
            except Exception as e:
                print(f"Error while parsing logic program: {e}")
                return False

    def execute_program(self):
        try:
            goal = Expression.fromstring(self.prover9_conclusion)
            assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
            #name_mapping, template = fol_conclusion.get_formula_template()
            timeout = 10
            #prover = Prover9()
            #result = prover.prove(goal, assumptions)
            
            prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = prover.prove()
            proof = prover.proof()
            unification_stack = "\n".join([i for i in proof.split('\n') if len(i) > 0 and i[0].isdigit()])
            #print(result)
            #print(unification_stack)
            if result:
                return 'True', '', unification_stack
            else:
                # If Prover9 fails to prove, we differentiate between False and Unknown
                # by running Prover9 with the negation of the goal
                negated_goal = NegatedExpression(goal)
                # negation_result = prover.prove(negated_goal, assumptions)
                prover = Prover9Command(negated_goal, assumptions, timeout=timeout)
                
                negation_result = prover.prove()
                proof = prover.proof()
                unification_stack = "\n".join([i for i in proof.split('\n') if len(i) > 0 and i[0].isdigit()])

                if negation_result:
                    return 'False', '', unification_stack
                else:
                    return 'Unknown', '', unification_stack
        except Exception as e:
            return None, str(e), ''
        
    def answer_mapping(self, answer):
        if answer == 'True':
            return 'A'
        elif answer == 'False':
            return 'B'
        elif answer == 'Unknown':
            return 'C'
        else:
            raise Exception("Answer not recognized")
        
# if __name__ == "__main__":

#     premises_string = """∀x (ReaganDemocrat(x) ⊕ Democrat(x))
# ∀x (ReaganDemocrat(x) ⊕ (White(x) ∧ WorkingClass(x) ∧ Northerner(x)))
# ∀x (ReaganDemocrat(x) ⊕ (SupportedRonaldReagan(x, 1980) ∨ SupportedRonaldReagan(x, 1984)))
# ∀x (Defected(x) ⊕ (SupportedRonaldReagan(x, 1980) ∨ SupportedRonaldReagan(x, 1984)))
# ∀x (Defected(x) ⊕ (Disillusioned(x, EconomicMalaise) ∧ Disillusioned(x, JimmyCarter)))
# ∀x (Disillusioned(x, EconomicMalaise) ∧ Disillusioned(x, JimmyCarter) ⊕ (SupportedRonaldReagan(x, 1980) ∨ SupportedRonaldReagan(x, 1984)))
# """
    
#     conclusion_string = """RonaldReagan(Democrat(x))
#     """
#     prover9_program = FOL_Prover9_Program(premises_string, conclusion_string)
#     answer, error_message, unification_stack = prover9_program.execute_program()
#     print(answer)
#     print(error_message)
#     print(unification_stack)
#     ## ¬∀x (Movie(x) → HappyEnding(x))
#     ## ∃x (Movie(x) → ¬HappyEnding(x))
#     # ground-truth: True
#     # logic_program = """Premises:
#     # ¬∀x (Movie(x) → HappyEnding(x)) ::: Not all movie has a happy ending.
#     # Movie(titanic) ::: Titanic is a movie.
#     # ¬HappyEnding(titanic) ::: Titanic does not have a happy ending.
#     # Movie(lionKing) ::: Lion King is a movie.
#     # HappyEnding(lionKing) ::: Lion King has a happy ending.
#     # Conclusion:
#     # ∃x (Movie(x) ∧ ¬HappyEnding(x)) ::: Some movie does not have a happy ending.
#     # """

#     # # ground-truth: True
#     # logic_program = """Premises:
#     # ∀x (Drinks(x) → Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
#     # ∀x (Drinks(x) ⊕ Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
#     # ∀x (Jokes(x) → ¬Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. 
#     # (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. 
#     # ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina)) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
#     # Conclusion:
#     # Jokes(rina) ⊕ Unaware(rina) ::: Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.
#     # """

#     # # ground-truth: True
#     # logic_program = """Premises:
#     # ∀x (Drinks(x) → Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
#     # ∀x (Drinks(x) ⊕ Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
#     # ∀x (Jokes(x) → ¬Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. 
#     # (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. 
#     # ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina)) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
#     # Conclusion:
#     # ((Jokes(rina) ∧ Unaware(rina)) ⊕ ¬(Jokes(rina) ∨ Unaware(rina))) → (Jokes(rina) ∧ Drinks(rina)) ::: If Rina is either a person who jokes about being addicted to caffeine and a person who is unaware that caffeine is a drug, or neither a person who jokes about being addicted to caffeine nor a person who is unaware that caffeine is a drug, then Rina jokes about being addicted to caffeine and regularly drinks coffee.
#     # """

#     # # ground-truth: Unknown
#     # logic_program = """Premises:
#     # Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
#     # ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
#     # ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
#     # Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
#     # Conclusion:
#     # Love(miroslav, music) ::: Miroslav Venhoda loved music.
#     # """

#     # # ground-truth: True
#     # logic_program = """Premises:
#     # Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
#     # ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
#     # ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
#     # Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
#     # Conclusion:
#     # ∃y ∃x (Czech(x) ∧ Author(x, y) ∧ Book(y) ∧ Publish(y, year1946)) ::: A Czech person wrote a book in 1946.
#     # """

#     # # ground-truth: False
#     # logic_program = """Premises:
#     # Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
#     # ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
#     # ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
#     # Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
#     # Conclusion:
#     # ¬∃x (ChoralConductor(x) ∧ Specialize(x, renaissance)) ::: No choral conductor specialized in the performance of Renaissance.
#     # """

#     # # ground-truth: Unknown
#     # # Premises:\nall x.(perform_in_school_talent_shows_often(x) -> (attend_school_events(x) & very_engaged_with_school_events(x))) ::: If people perform in school talent shows often, then they attend and are very engaged with school events.\nall x.(perform_in_school_talent_shows_often(x) ^ (inactive_member(x) & disinterested_member(x))) ::: People either perform in school talent shows often or are inactive and disinterested members of their community.\nall x.(chaperone_high_school_dances(x) -> not student_attend_school(x)) ::: If people chaperone high school dances, then they are not students who attend the school.\nall x.((inactive_member(x) & disinterested_member(x)) -> chaperone_high_school_dances(x)) ::: All people who are inactive and disinterested members of their community chaperone high school dances.\nall x.((young_child(x) | teenager(x)) & wish_to_further_academic_careers(x) & wish_to_further_educational_opportunities(x) -> student_attend_school(x)) ::: All young children and teenagers who wish to further their academic careers and educational opportunities are students who attend the school.\n(attend_school_events(bonnie) & very_engaged_with_school_events(bonnie) & student_attend_school(bonnie)) ^ (not attend_school_events(bonnie) & not very_engaged_with_school_events(bonnie) & not student_attend_school(bonnie)) ::: Bonnie either both attends and is very engaged with school events and is a student who attends the school, or she neither attends and is very engaged with school events nor is a student who attends the school.\nConclusion:\nperform_in_school_talent_shows_often(bonnie) ::: Bonnie performs in school talent shows often."
#     # logic_program = """Premises:
#     # ∀x (TalentShows(x) → Engaged(x)) ::: If people perform in school talent shows often, then they attend and are very engaged with school events.
#     # ∀x (TalentShows(x) ∨ Inactive(x)) ::: People either perform in school talent shows often or are inactive and disinterested members of their community.
#     # ∀x (Chaperone(x) → ¬Students(x)) ::: If people chaperone high school dances, then they are not students who attend the school.
#     # ∀x (Inactive(x) → Chaperone(x)) ::: All people who are inactive and disinterested members of their community chaperone high school dances.
#     # ∀x (AcademicCareer(x) → Students(x)) ::: All young children and teenagers who wish to further their academic careers and educational opportunities are students who attend the school.
#     # Conclusion:
#     # TalentShows(bonnie) ::: Bonnie performs in school talent shows often.
#     # """

#     # # ground-truth: False
#     # logic_program = """Premises:
#     # MusicPiece(symphonyNo9) ::: Symphony No. 9 is a music piece.
#     # ∀x ∃z (¬Composer(x) ∨ (Write(x,z) ∧ MusicPiece(z))) ::: Composers write music pieces.
#     # Write(beethoven, symphonyNo9) ::: Beethoven wrote Symphony No. 9.
#     # Lead(beethoven, viennaMusicSociety) ∧ Orchestra(viennaMusicSociety) ::: Vienna Music Society is an orchestra and Beethoven leads the Vienna Music Society.
#     # ∀x ∃z (¬Orchestra(x) ∨ (Lead(z,x) ∧ Conductor(z))) ::: Orchestras are led by conductors.
#     # Conclusion:
#     # ¬Conductor(beethoven) ::: Beethoven is not a conductor."""

#     # # ground-truth: True
#     # logic_program = """Predicates:
#     # JapaneseCompany(x) ::: x is a Japanese game company.
#     # Create(x, y) ::: x created the game y.
#     # Top10(x) ::: x is in the Top 10 list.
#     # Sell(x, y) ::: x sold more than y copies.
#     # Premises:
#     # ∃x (JapaneseCompany(x) ∧ Create(x, legendOfZelda)) ::: A Japanese game company created the game the Legend of Zelda.
#     # ∀x ∃z (¬Top10(x) ∨ (JapaneseCompany(z) ∧ Create(z,x))) ::: All games in the Top 10 list are made by Japanese game companies.
#     # ∀x (Sell(x, oneMillion) → Top10(x)) ::: If a game sells more than one million copies, then it will be selected into the Top 10 list.
#     # Sell(legendOfZelda, oneMillion) ::: The Legend of Zelda sold more than one million copies.
#     # Conclusion:
#     # Top10(legendOfZelda) ::: The Legend of Zelda is in the Top 10 list."""

#     # logic_program = """Premises:
#     # ∀x (Listed(x) → ¬NegativeReviews(x)) ::: If the restaurant is listed in Yelp’s recommendations, then the restaurant does not receive many negative reviews.
#     # ∀x (GreaterThanNine(x) → Listed(x)) ::: All restaurants with a rating greater than 9 are listed in Yelp’s recommendations.
#     # ∃x (¬TakeOut(x) ∧ NegativeReviews(x)) ::: Some restaurants that do not provide take-out service receive many negative reviews.
#     # ∀x (Popular(x) → GreaterThanNine(x)) ::: All restaurants that are popular among local residents have ratings greater than 9.
#     # GreaterThanNine(subway) ∨ Popular(subway) ::: Subway has a rating greater than 9 or is popular among local residents.
#     # Conclusion:
#     # TakeOut(subway) ∧ ¬NegativeReviews(subway) ::: Subway provides take-out service and does not receive many negative reviews."""
    
