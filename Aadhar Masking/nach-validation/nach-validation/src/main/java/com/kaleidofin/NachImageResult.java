package com.kaleidofin;

public class NachImageResult {
	 boolean rectangleDetected;

	    String accountNumber;

	    String ifscCode;

	    String formNumber;

	    boolean monthlyTicked;

	    boolean quarterlyTicked;

	    boolean halfYearlyTicked;

	    boolean yearlyTicked;

	    boolean fixedAmountTicked;

	    boolean debitTicked;

	    boolean bankNamePresent;

	    boolean namePresent;

	    boolean signaturePresent;

	    float timeTaken;

	    public String getAccountNumber() {
	        return accountNumber;
	    }

	    public void setAccountNumber(String accountNumber) {
	        this.accountNumber = accountNumber;
	    }

	    public String getIfscCode() {
	        return ifscCode;
	    }

	    public void setIfscCode(String ifscCode) {
	        this.ifscCode = ifscCode;
	    }

	    public String getFormNumber() {
	        return formNumber;
	    }

	    public void setFormNumber(String formNumber) {
	        this.formNumber = formNumber;
	    }

	    public boolean isMonthlyTicked() {
	        return monthlyTicked;
	    }

	    public void setMonthlyTicked(boolean monthlyTicked) {
	        this.monthlyTicked = monthlyTicked;
	    }

	    public boolean isQuarterlyTicked() {
	        return quarterlyTicked;
	    }

	    public void setQuarterlyTicked(boolean quarterlyTicked) {
	        this.quarterlyTicked = quarterlyTicked;
	    }

	    public boolean isHalfYearlyTicked() {
	        return halfYearlyTicked;
	    }

	    public void setHalfYearlyTicked(boolean halfYearlyTicked) {
	        this.halfYearlyTicked = halfYearlyTicked;
	    }

	    public boolean isYearlyTicked() {
	        return yearlyTicked;
	    }

	    public void setYearlyTicked(boolean yearlyTicked) {
	        this.yearlyTicked = yearlyTicked;
	    }

	    public boolean isFixedAmountTicked() {
	        return fixedAmountTicked;
	    }

	    public void setFixedAmountTicked(boolean fixedAmountTicked) {
	        this.fixedAmountTicked = fixedAmountTicked;
	    }

	    public boolean isDebitTicked() {
	        return debitTicked;
	    }

	    public void setDebitTicked(boolean debitTicked) {
	        this.debitTicked = debitTicked;
	    }

	    public boolean isBankNamePresent() {
	        return bankNamePresent;
	    }

	    public void setBankNamePresent(boolean bankNamePresent) {
	        this.bankNamePresent = bankNamePresent;
	    }

	    public boolean isNamePresent() {
	        return namePresent;
	    }

	    public void setNamePresent(boolean namePresent) {
	        this.namePresent = namePresent;
	    }

	    public boolean isSignaturePresent() {
	        return signaturePresent;
	    }

	    public void setSignaturePresent(boolean signaturePresent) {
	        this.signaturePresent = signaturePresent;
	    }

	    public boolean isRectangleDetected() {
	        return rectangleDetected;
	    }

	    public void setRectangleDetected(boolean rectangleDetected) {
	        this.rectangleDetected = rectangleDetected;
	    }

	    public float getTimeTaken() {
	        return timeTaken;
	    }

	    public void setTimeTaken(float timeTaken) {
	        this.timeTaken = timeTaken;
	    }

	    @Override
	    public String toString() {
	        return "NachImageResult{" +
	                "rectangleDetected=" + rectangleDetected +
	                ", accountNumber='" + accountNumber + '\'' +
	                ", ifscCode='" + ifscCode + '\'' +
	                ", formNumber='" + formNumber + '\'' +
	                ", monthlyTicked=" + monthlyTicked +
	                ", quarterlyTicked=" + quarterlyTicked +
	                ", halfYearlyTicked=" + halfYearlyTicked +
	                ", yearlyTicked=" + yearlyTicked +
	                ", fixedAmountTicked=" + fixedAmountTicked +
	                ", debitTicked=" + debitTicked +
	                ", bankNamePresent=" + bankNamePresent +
	                ", namePresent=" + namePresent +
	                ", signaturePresent=" + signaturePresent +
	                ", timeTaken=" + timeTaken +
	                '}';
	    }

}
